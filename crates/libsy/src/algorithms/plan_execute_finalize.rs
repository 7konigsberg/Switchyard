// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plans with a capable model, executes with an efficient model, then gives final control back.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{
    Category, ContentBlock, InstructionBlock, Message, Request, Role, WireFormat,
};

use super::advisor_gate::turn::{buffer_turn, has_tool_use, visible_text};
use super::advisor_gate::{extend_exact_responses_turn, response_message};
use super::plan_execute::{DEFAULT_PLANNING_PROMPT, ExecutionTracker};
use super::util::prompts::{drop_exact_replay, prepend_system_prompt};
use super::util::tool_signals::is_mutating_tool_call;
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::{LibsyError, Result, RoutingOutcome};

/// Terminal response pattern validated against Codex DeepSWE trajectories.
pub const DEFAULT_FINALIZER_TERMINAL_PATTERN: &str =
    r"(?i)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:implemented|completed|done|erledigt|fertig)(?:\*\*)?\b";

/// Instruction used after the efficient executor declares completion.
pub const DEFAULT_FINALIZER_PROMPT: &str =
    include_str!("../prompts/plan-execute/finalizer-system-prompt.md");

const EMPTY_COMPLETION: &str = "(the efficient executor produced no visible completion)";
const MAX_SESSION_STATE: usize = 4_096;

/// Configuration for [`PlanExecuteFinalize`].
#[derive(Clone, Debug)]
pub struct PlanExecuteFinalizeConfig {
    /// System instruction prepended while the capable model plans.
    pub planning_prompt: String,
    /// System instruction added when the capable model takes final control.
    pub finalizer_prompt: String,
    /// Pattern that identifies an executor completion response.
    pub terminal_pattern: String,
}

impl Default for PlanExecuteFinalizeConfig {
    fn default() -> Self {
        Self {
            planning_prompt: DEFAULT_PLANNING_PROMPT.trim().to_string(),
            finalizer_prompt: DEFAULT_FINALIZER_PROMPT.trim().to_string(),
            terminal_pattern: DEFAULT_FINALIZER_TERMINAL_PATTERN.to_string(),
        }
    }
}

/// Routes planning and finalization to the capable model, with efficient execution between them.
pub struct PlanExecuteFinalize {
    planning_prompt: String,
    finalizer_prompt: String,
    finalizer_system_prompt: String,
    terminal_pattern: regex::Regex,
    execution: ExecutionTracker,
    finalizing: Mutex<HashSet<RoutingIdentity>>,
    planner_checkpoints: Mutex<HashMap<RoutingIdentity, PlannerCheckpoint>>,
}

#[derive(Clone)]
struct PlannerCheckpoint {
    messages: Vec<Message>,
    responses_input: Option<Vec<serde_json::Value>>,
}

impl PlanExecuteFinalize {
    /// Creates a plan, execute, and capable-finalize router.
    pub fn new(config: PlanExecuteFinalizeConfig) -> Result<Self> {
        if config.planning_prompt.trim().is_empty() {
            return Err(algorithm_error("planning_prompt must not be empty"));
        }
        if config.finalizer_prompt.trim().is_empty() {
            return Err(algorithm_error("finalizer_prompt must not be empty"));
        }
        if config.terminal_pattern.is_empty() {
            return Err(algorithm_error("terminal_pattern must not be empty"));
        }
        let terminal_pattern = regex::Regex::new(&config.terminal_pattern)
            .map_err(|error| algorithm_error(format!("terminal_pattern is invalid: {error}")))?;
        let finalizer_system_prompt = format!(
            "{}\n\n{}",
            config.planning_prompt.trim(),
            config.finalizer_prompt.trim()
        );
        Ok(Self {
            planning_prompt: config.planning_prompt,
            finalizer_prompt: config.finalizer_prompt,
            finalizer_system_prompt,
            terminal_pattern,
            execution: ExecutionTracker::new(),
            finalizing: Mutex::new(HashSet::new()),
            planner_checkpoints: Mutex::new(HashMap::new()),
        })
    }

    fn is_finalizing(&self, request: &Request) -> bool {
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return false;
        };
        let mut sessions = self.finalizing.lock();
        let active = sessions.contains(&identity);
        if request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true)
        {
            sessions.remove(&identity);
            self.planner_checkpoints.lock().remove(&identity);
        }
        active
    }

    fn begin_finalizing(&self, request: &Request) -> Result<()> {
        let identity = RoutingIdentity::from_request(request)
            .ok_or_else(|| algorithm_error("plan_execute_finalize requires a session identity"))?;
        let mut sessions = self.finalizing.lock();
        if sessions.len() >= MAX_SESSION_STATE
            && !sessions.contains(&identity)
            && let Some(evicted) = sessions.iter().next().cloned()
        {
            sessions.remove(&evicted);
            self.planner_checkpoints.lock().remove(&evicted);
        }
        sessions.insert(identity);
        Ok(())
    }

    fn execution_context(&self, request: &Request) -> Request {
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return request.clone();
        };
        let request_has_mutation = request
            .llm_request
            .messages
            .iter()
            .any(message_has_mutation);
        let mut checkpoints = self.planner_checkpoints.lock();
        if request_has_mutation && !checkpoints.contains_key(&identity) {
            if checkpoints.len() >= MAX_SESSION_STATE
                && let Some(evicted) = checkpoints.keys().next().cloned()
            {
                checkpoints.remove(&evicted);
            }
            checkpoints.insert(identity.clone(), planner_checkpoint(request));
        }
        checkpoints.get(&identity).map_or_else(
            || request.clone(),
            |prefix| {
                if request_starts_with(prefix, request) {
                    request.clone()
                } else {
                    restore_planner_prefix(prefix, request)
                }
            },
        )
    }

    fn prepare_finalizer_request(
        &self,
        request: &Request,
        completion: &switchyard_protocol::AggLlmResponse,
    ) -> Request {
        let mut finalizer = self.execution_context(request);
        let echo = response_message(completion, EMPTY_COMPLETION);
        finalizer.llm_request.messages.push(echo);
        let exact_extended =
            extend_exact_responses_turn(&mut finalizer, completion, &self.finalizer_prompt);
        finalizer
            .llm_request
            .messages
            .push(Message::text(Role::User, self.finalizer_prompt.clone()));
        prepend_system_prompt_preserving_responses(&mut finalizer, &self.finalizer_system_prompt);
        if !exact_extended {
            drop_exact_replay(&mut finalizer);
        }
        finalizer
    }

    fn prepare_continuation(&self, request: &Request) -> Request {
        let mut continued = self.execution_context(request);
        prepend_system_prompt_preserving_responses(&mut continued, &self.finalizer_system_prompt);
        continued
    }
}

#[async_trait::async_trait]
impl Algorithm for PlanExecuteFinalize {
    fn name(&self) -> &str {
        "plan_execute_finalize"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        if self.is_finalizing(&request) {
            let capable = driver.models_for(&Category::Capable);
            let request = self.prepare_continuation(&request);
            return route_first(capable, request, "finalize");
        }

        if !self.execution.is_executing(&request) {
            prepend_system_prompt(&mut request, &self.planning_prompt);
            return route_first(driver.models_for(&Category::Capable), request, "plan");
        }

        let efficient = driver.models_for(&Category::Efficient).to_vec();
        let selected = efficient
            .first()
            .ok_or_else(|| algorithm_error("no models available for category Efficient"))?
            .clone();
        let response = driver.call_model(request.clone(), efficient).await?;
        let served = response
            .served_model()
            .cloned()
            .unwrap_or_else(|| selected.clone());
        let turn = buffer_turn(served.as_str(), response).await?;
        let terminal = !has_tool_use(&turn.agg)
            && visible_text(&turn.agg).is_some_and(|text| self.terminal_pattern.is_match(&text));
        if !terminal {
            return Ok(RoutingOutcome::answered(
                served,
                request,
                turn.into_response(),
            ));
        }

        self.begin_finalizing(&request)?;
        driver.set_evidence(serde_json::json!({
            "source": "plan_execute_finalize",
            "verdict": "finalize",
        }));
        let finalizer_request = self.prepare_finalizer_request(&request, &turn.agg);
        route_first(
            driver.models_for(&Category::Capable),
            finalizer_request,
            "finalize",
        )
    }
}

fn route_first(
    models: &[switchyard_protocol::ModelId],
    request: Request,
    phase: &str,
) -> Result<RoutingOutcome> {
    let selected = models
        .first()
        .ok_or_else(|| algorithm_error(format!("no models available for {phase} phase")))?
        .clone();
    tracing::info!(target = %selected, phase, "plan-execute-finalize selected target");
    Ok(RoutingOutcome::route_to(
        selected,
        models[1..].to_vec(),
        request,
    ))
}

fn planner_checkpoint(request: &Request) -> PlannerCheckpoint {
    PlannerCheckpoint {
        messages: request.llm_request.messages.clone(),
        responses_input: exact_responses_input(request).cloned(),
    }
}

fn restore_planner_prefix(prefix: &PlannerCheckpoint, current: &Request) -> Request {
    let mut restored = current.clone();
    restored.llm_request.messages = prefix
        .messages
        .iter()
        .cloned()
        .chain(current.llm_request.messages.iter().cloned())
        .collect();
    let current_input = exact_responses_input_mut(&mut restored);
    if let (Some(prefix_input), Some(current_input)) = (&prefix.responses_input, current_input) {
        *current_input = prefix_input
            .iter()
            .cloned()
            .chain(current_input.iter().cloned())
            .collect();
    }
    restored
}

fn request_starts_with(prefix: &PlannerCheckpoint, current: &Request) -> bool {
    if let (Some(prefix_input), Some(current_input)) =
        (&prefix.responses_input, exact_responses_input(current))
    {
        return !prefix_input.is_empty() && current_input.starts_with(prefix_input);
    }
    !prefix.messages.is_empty() && current.llm_request.messages.starts_with(&prefix.messages)
}

fn exact_responses_input(request: &Request) -> Option<&Vec<serde_json::Value>> {
    request
        .llm_request
        .preservation
        .requests
        .get(&switchyard_protocol::FormatId::known(
            WireFormat::OpenAiResponses,
        ))?
        .get("input")?
        .as_array()
}

fn exact_responses_input_mut(request: &mut Request) -> Option<&mut Vec<serde_json::Value>> {
    request
        .llm_request
        .preservation
        .requests
        .get_mut(&switchyard_protocol::FormatId::known(
            WireFormat::OpenAiResponses,
        ))?
        .get_mut("input")?
        .as_array_mut()
}

fn prepend_system_prompt_preserving_responses(request: &mut Request, prompt: &str) {
    let has_prompt = request
        .llm_request
        .instructions
        .first()
        .is_some_and(|block| {
            block.role == Role::System
                && block.content.first().is_some_and(
                    |content| matches!(content, ContentBlock::Text { text } if text == prompt),
                )
        });
    if !has_prompt {
        request.llm_request.instructions.insert(
            0,
            InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: prompt.to_string(),
                }],
            },
        );
    }
    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    let Some(body) = request
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(serde_json::Value::as_object_mut)
    else {
        drop_exact_replay(request);
        return;
    };
    let instructions = body
        .get("instructions")
        .and_then(serde_json::Value::as_str)
        .filter(|instructions| !instructions.is_empty())
        .map_or_else(
            || prompt.to_string(),
            |instructions| {
                if instructions == prompt || instructions.starts_with(&format!("{prompt}\n\n")) {
                    instructions.to_string()
                } else {
                    format!("{prompt}\n\n{instructions}")
                }
            },
        );
    body.insert(
        "instructions".to_string(),
        serde_json::Value::String(instructions),
    );
    request
        .llm_request
        .preservation
        .requests
        .retain(|candidate, _| candidate == &format);
}

fn message_has_mutation(message: &Message) -> bool {
    message.content.iter().any(|block| {
        matches!(block, ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments))
    })
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use switchyard_protocol::{
        ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolCall, ToolResult,
    };

    use super::*;
    use crate::core::algorithm::RuntimeModels;
    use crate::core::testing::{reply, test_drive_with_models};

    fn algorithm() -> Arc<dyn Algorithm> {
        Arc::new(
            PlanExecuteFinalize::new(PlanExecuteFinalizeConfig::default())
                .expect("default config is valid"),
        )
    }

    fn models() -> RuntimeModels {
        HashMap::from([
            (Category::Capable, vec![ModelId::from("model/sol")]),
            (Category::Efficient, vec![ModelId::from("model/luna")]),
        ])
        .into()
    }

    fn request(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/finalize".to_string()),
                messages,
                ..LlmRequest::default()
            },
            metadata: Some(Metadata {
                session_id: Some("task-1".to_string()),
                ..Metadata::default()
            }),
            ..Request::default()
        }
    }

    fn edit_and_result() -> Vec<Message> {
        vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "edit-1".to_string(),
                    name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"patch": "change"}),
                })],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "edit-1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "done".to_string(),
                    }],
                    is_error: Some(false),
                })],
            },
        ]
    }

    #[tokio::test]
    async fn terminal_executor_turn_hands_tools_to_capable_finalizer() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            algorithm(),
            request([vec![Message::text(Role::User, "fix it")], edit_and_result()].concat()),
            models(),
            move |model: ModelId, request: Request| {
                captured
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async move {
                    if model.as_str() == "model/luna" {
                        Ok(reply("Completed the implementation"))
                    } else {
                        Ok(reply("finalized"))
                    }
                }
            },
        )
        .await
        .expect("finalization routes");

        assert_eq!(selected.as_str(), "model/sol");
        let calls = calls.lock().expect("call log is available");
        assert_eq!(
            calls
                .iter()
                .map(|(model, _)| model.as_str())
                .collect::<Vec<_>>(),
            vec!["model/luna", "model/sol"]
        );
        let finalizer = &calls[1].1.llm_request;
        assert!(finalizer.tool_choice != Some(switchyard_protocol::ToolChoice::None));
        assert!(
            finalizer
                .messages
                .iter()
                .filter_map(|message| message.text_content("\n"))
                .any(|text| text.contains("Completed the implementation"))
        );
        assert!(
            finalizer
                .instructions
                .first()
                .and_then(|block| block.content.first())
                .is_some_and(|block| matches!(block, ContentBlock::Text { text } if text.contains("live repository")))
        );
    }

    #[tokio::test]
    async fn finalizer_keeps_control_after_a_tool_call() {
        let algorithm = algorithm();
        let first_calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&first_calls);
        test_drive_with_models(
            Arc::clone(&algorithm),
            request([vec![Message::text(Role::User, "fix it")], edit_and_result()].concat()),
            models(),
            move |model: ModelId, request: Request| {
                captured
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async move { Ok(reply("Completed the implementation")) }
            },
        )
        .await
        .expect("finalization starts");

        let finalizer_tool = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "test-1".to_string(),
                name: "exec_command".to_string(),
                arguments: serde_json::json!({"cmd": "git diff && cargo test"}),
            })],
        };
        let result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "test-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "tests pass".to_string(),
                }],
                is_error: Some(false),
            })],
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let continued = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            algorithm,
            request(
                [
                    vec![Message::text(Role::User, "fix it")],
                    edit_and_result(),
                    vec![finalizer_tool, result],
                ]
                .concat(),
            ),
            models(),
            move |model: ModelId, request: Request| {
                continued
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async { Ok(reply("done")) }
            },
        )
        .await
        .expect("finalization continues");

        assert_eq!(selected.as_str(), "model/sol");
        assert_eq!(
            calls.lock().expect("call log is available")[0].0,
            "model/sol"
        );
    }

    #[tokio::test]
    async fn nonterminal_executor_turn_is_replayed_without_finalizing() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            algorithm(),
            request([vec![Message::text(Role::User, "fix it")], edit_and_result()].concat()),
            models(),
            move |model: ModelId, request: Request| {
                captured
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async { Ok(reply("I am still testing")) }
            },
        )
        .await
        .expect("execution routes");

        assert_eq!(selected.as_str(), "model/luna");
        assert_eq!(calls.lock().expect("call log is available").len(), 1);
    }
}
