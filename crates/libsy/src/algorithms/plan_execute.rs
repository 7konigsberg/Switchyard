// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Starts coding tasks on a capable planner, then hands execution to an efficient model.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, FormatId, LlmResponse, Message, ModelId, Request, Response, Role,
    ToolCall, ToolChoice, WireFormat,
};

use super::util::prompts::{SystemPromptProcessor, TargetPrompts, append_note, drop_exact_replay};
use super::util::tool_signals::{ToolSignals, is_mutating_tool_call};
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::processor::{Event, Processor};
use crate::{LibsyError, Result, RoutingOutcome};

/// Default instruction prepended while the capable model is planning.
pub const DEFAULT_PLANNING_PROMPT: &str =
    include_str!("../prompts/plan-execute/planning-system-prompt.md");

/// Default instruction used when the planner re-enters during execution.
pub const DEFAULT_CHECKPOINT_PROMPT: &str =
    include_str!("../prompts/plan-execute/checkpoint-system-prompt.md");

/// Maximum session latches retained by one router instance.
const MAX_EXECUTING_SESSIONS: usize = 4_096;
/// Tool call pairs retained after the initial handoff for checkpoint context.
const RECENT_CHECKPOINT_TOOL_CALLS: usize = 6;

/// Configuration for [`PlanExecute`].
#[derive(Clone, Debug)]
pub struct PlanExecuteConfig {
    /// System instruction prepended until the first edit or write tool call.
    pub planning_prompt: String,
    /// Planner instruction appended during recurring execution checkpoints.
    pub checkpoint_prompt: String,
    /// Efficient-model turns between checkpoints. Zero disables checkpoints.
    pub checkpoint_interval_turns: u32,
    /// Maximum checkpoints opened during one session.
    pub max_checkpoints: u32,
    /// Maximum capable-model turns within one checkpoint.
    pub max_checkpoint_turns: u32,
    /// Maximum capable-model checkpoint turns across one session.
    pub max_checkpoint_turns_total: u32,
}

impl Default for PlanExecuteConfig {
    fn default() -> Self {
        Self {
            planning_prompt: DEFAULT_PLANNING_PROMPT.trim().to_string(),
            checkpoint_prompt: DEFAULT_CHECKPOINT_PROMPT.trim().to_string(),
            checkpoint_interval_turns: 0,
            max_checkpoints: 8,
            max_checkpoint_turns: 4,
            max_checkpoint_turns_total: 16,
        }
    }
}

/// Routes planning turns to a capable model and all turns after the first edit
/// to an efficient model while preserving the caller's full trajectory.
pub struct PlanExecute {
    capable: ModelId,
    efficient: ModelId,
    planning_prompt: SystemPromptProcessor,
    checkpoint_prompt: String,
    checkpoint_interval_turns: u32,
    max_checkpoints: u32,
    max_checkpoint_turns: u32,
    max_checkpoint_turns_total: u32,
    sessions: Mutex<HashMap<RoutingIdentity, ExecutionState>>,
}

#[derive(Default)]
struct ExecutionState {
    efficient_turns: u32,
    checkpoints: u32,
    checkpoint_turns: Option<u32>,
    checkpoint_turns_total: u32,
    checkpoint_known_mutations: HashSet<String>,
}

enum Phase {
    Plan,
    Execute,
    Checkpoint { force_handoff: bool },
}

impl PlanExecute {
    /// Creates a plan/execute router.
    ///
    /// Returns an error when the planning prompt is empty.
    pub fn new(capable: ModelId, efficient: ModelId, config: PlanExecuteConfig) -> Result<Self> {
        if config.planning_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "planning_prompt must not be empty".to_string(),
            });
        }
        if config.checkpoint_interval_turns > 0 && config.checkpoint_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "checkpoint_prompt must not be empty when checkpoints are enabled"
                    .to_string(),
            });
        }
        if config.checkpoint_interval_turns > 0
            && (config.max_checkpoints == 0
                || config.max_checkpoint_turns == 0
                || config.max_checkpoint_turns_total == 0)
        {
            return Err(LibsyError::AlgorithmError {
                message: "checkpoint limits must be at least 1 when checkpoints are enabled"
                    .to_string(),
            });
        }
        let planning_prompt = SystemPromptProcessor::new(
            TargetPrompts::default().with(capable.clone(), config.planning_prompt),
        );
        Ok(Self {
            capable,
            efficient,
            planning_prompt,
            checkpoint_prompt: config.checkpoint_prompt,
            checkpoint_interval_turns: config.checkpoint_interval_turns,
            max_checkpoints: config.max_checkpoints,
            max_checkpoint_turns: config.max_checkpoint_turns,
            max_checkpoint_turns_total: config.max_checkpoint_turns_total,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// Chooses the next phase and advances its bounded per-session counters.
    fn phase(&self, request: &Request) -> Phase {
        let signals = ToolSignals::from_request(request, None);
        let mutation_seen = signals.edit_count > 0 || signals.write_count > 0;
        let mutation_ids = mutation_call_ids(request);
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return if mutation_seen {
                Phase::Execute
            } else {
                Phase::Plan
            };
        };

        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let mut sessions = self.sessions.lock();
        if session_final {
            let was_executing = sessions.remove(&identity).is_some();
            return if mutation_seen || was_executing {
                Phase::Execute
            } else {
                Phase::Plan
            };
        }

        if mutation_seen && !sessions.contains_key(&identity) {
            if sessions.len() >= MAX_EXECUTING_SESSIONS
                && let Some(evicted) = sessions.keys().next().cloned()
            {
                sessions.remove(&evicted);
            }
            sessions.insert(identity.clone(), ExecutionState::default());
        }

        let Some(state) = sessions.get_mut(&identity) else {
            return Phase::Plan;
        };
        if let Some(turns) = state.checkpoint_turns.as_mut() {
            if mutation_ids
                .iter()
                .any(|call_id| !state.checkpoint_known_mutations.contains(call_id))
            {
                state.checkpoint_turns = None;
                state.checkpoint_known_mutations.clear();
                state.efficient_turns = 1;
                return Phase::Execute;
            }
            *turns = turns.saturating_add(1);
            state.checkpoint_turns_total = state.checkpoint_turns_total.saturating_add(1);
            return Phase::Checkpoint {
                force_handoff: *turns >= self.max_checkpoint_turns
                    || state.checkpoint_turns_total >= self.max_checkpoint_turns_total,
            };
        }
        if self.checkpoint_interval_turns > 0
            && state.efficient_turns >= self.checkpoint_interval_turns
            && state.checkpoints < self.max_checkpoints
            && state.checkpoint_turns_total < self.max_checkpoint_turns_total
        {
            state.checkpoints += 1;
            state.checkpoint_turns = Some(1);
            state.checkpoint_turns_total += 1;
            state.checkpoint_known_mutations = mutation_ids;
            return Phase::Checkpoint {
                force_handoff: self.max_checkpoint_turns == 1
                    || state.checkpoint_turns_total >= self.max_checkpoint_turns_total,
            };
        }
        state.efficient_turns = state.efficient_turns.saturating_add(1);
        Phase::Execute
    }

    fn finish_checkpoint(&self, request: &Request) {
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return;
        };
        if let Some(state) = self.sessions.lock().get_mut(&identity) {
            state.checkpoint_turns = None;
            state.checkpoint_known_mutations.clear();
            state.efficient_turns = 1;
        }
    }

    async fn checkpoint(
        &self,
        driver: &Driver,
        mut request: Request,
        force_handoff: bool,
    ) -> Result<RoutingOutcome> {
        let downstream_stream = request.llm_request.stream;
        let mut checkpoint_request = compact_execution_context(&request);
        checkpoint_request.llm_request.stream = true;
        // Match the initial planning path so their shared prefix stays cacheable.
        self.planning_prompt
            .process(
                &mut (),
                Event::Decision {
                    request: &mut checkpoint_request,
                    selected_model_id: &self.capable,
                },
            )
            .await?;
        append_note(&mut checkpoint_request, &self.checkpoint_prompt);
        if force_handoff {
            checkpoint_request.llm_request.tool_choice = Some(ToolChoice::None);
        }

        let response = driver
            .call_model(checkpoint_request.clone(), vec![self.capable.clone()])
            .await?;
        let metadata = response.metadata;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|source| LibsyError::client_call(self.capable.clone(), source))?;
        let has_mutations = !mutating_calls(&aggregate).is_empty();
        let has_tools = aggregate.outputs.iter().any(|output| {
            output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
        });

        if has_mutations || has_tools && !force_handoff {
            let llm_response = if downstream_stream {
                LlmResponse::Stream(aggregate.into_stream())
            } else {
                LlmResponse::Agg(aggregate)
            };
            tracing::info!(
                target = %self.capable,
                phase = if has_mutations { "checkpoint_edit" } else { "checkpoint_probe" },
                "plan-execute selected target"
            );
            return Ok(RoutingOutcome::answered(
                self.capable.clone(),
                checkpoint_request,
                Response {
                    llm_response,
                    metadata,
                },
            ));
        }

        let handoff = checkpoint_handoff(&aggregate, force_handoff && has_tools);
        self.finish_checkpoint(&request);
        append_note_preserving_responses(&mut request, &handoff);
        tracing::info!(
            target = %self.efficient,
            phase = "checkpoint_handoff",
            "plan-execute selected target"
        );
        Ok(RoutingOutcome::route_to(
            self.efficient.clone(),
            Vec::new(),
            request,
        ))
    }
}

#[async_trait::async_trait]
impl Algorithm for PlanExecute {
    fn name(&self) -> &str {
        "plan_execute"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        match self.phase(&request) {
            Phase::Execute => {
                tracing::info!(target = %self.efficient, phase = "execute", "plan-execute selected target");
                Ok(RoutingOutcome::route_to(
                    self.efficient.clone(),
                    Vec::new(),
                    request,
                ))
            }
            Phase::Checkpoint { force_handoff } => {
                self.checkpoint(&driver, request, force_handoff).await
            }
            Phase::Plan => {
                self.planning_prompt
                    .process(
                        &mut (),
                        Event::Decision {
                            request: &mut request,
                            selected_model_id: &self.capable,
                        },
                    )
                    .await?;
                tracing::info!(target = %self.capable, phase = "plan", "plan-execute selected target");
                Ok(RoutingOutcome::route_to(
                    self.capable.clone(),
                    Vec::new(),
                    request,
                ))
            }
        }
    }
}

fn mutating_calls(response: &AggLlmResponse) -> Vec<&ToolCall> {
    response
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments) => {
                Some(call)
            }
            _ => None,
        })
        .collect()
}

fn mutation_call_ids(request: &Request) -> HashSet<String> {
    request
        .llm_request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments) => {
                Some(call.id.clone())
            }
            _ => None,
        })
        .collect()
}

fn checkpoint_handoff(response: &AggLlmResponse, blocked_by_budget: bool) -> String {
    let visible = response
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if blocked_by_budget {
        return format!(
            "The bounded planning checkpoint ended before its requested tools ran. Continue implementation using the evidence already available.\n\nLatest planning note:\n{}",
            if visible.is_empty() {
                "No additional explanation was provided."
            } else {
                &visible
            }
        );
    }
    format!(
        "A planning checkpoint inspected the current environment. Continue implementation using its latest assessment.\n\nUpdated planning note:\n{}",
        if visible.is_empty() {
            "The existing direction remains appropriate."
        } else {
            &visible
        }
    )
}

fn append_note_preserving_responses(request: &mut Request, note: &str) {
    match request.llm_request.messages.last_mut() {
        Some(last) if last.role == Role::User => last.content.push(ContentBlock::Text {
            text: note.to_string(),
        }),
        _ => request
            .llm_request
            .messages
            .push(Message::text(Role::User, note)),
    }
    let format = FormatId::known(WireFormat::OpenAiResponses);
    let Some(input) = request
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(|body| body.get_mut("input"))
        .and_then(serde_json::Value::as_array_mut)
    else {
        drop_exact_replay(request);
        return;
    };
    input.push(serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": note}],
    }));
    request
        .llm_request
        .preservation
        .requests
        .retain(|candidate, _| candidate == &format);
}

fn compact_execution_context(base: &Request) -> Request {
    let mut request = base.clone();
    request.llm_request.messages = compact_messages(&base.llm_request.messages);
    let format = FormatId::known(WireFormat::OpenAiResponses);
    if let Some(input) = request
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(|body| body.get_mut("input"))
        .and_then(serde_json::Value::as_array_mut)
    {
        *input = compact_responses_input(input);
    }
    request
}

fn compact_messages(messages: &[Message]) -> Vec<Message> {
    let Some(boundary) = messages.iter().position(message_has_mutation) else {
        return messages.to_vec();
    };
    let selected_call_ids = selected_message_call_ids(&messages[boundary..]);
    messages[..boundary]
        .iter()
        .cloned()
        .chain(messages[boundary..].iter().filter_map(|message| {
            let selected_call_in_message = message.content.iter().any(|block| {
                matches!(block, ContentBlock::ToolCall(call) if selected_call_ids.contains(&call.id))
            });
            let content = message
                .content
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolCall(call) => selected_call_ids.contains(&call.id),
                    ContentBlock::ToolResult(result) => {
                        selected_call_ids.contains(&result.tool_call_id)
                    }
                    _ => selected_call_in_message,
                })
                .cloned()
                .collect::<Vec<_>>();
            (!content.is_empty()).then_some(Message {
                role: message.role,
                content,
            })
        }))
        .collect()
}

fn message_has_mutation(message: &Message) -> bool {
    message.content.iter().any(|block| {
        matches!(block, ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments))
    })
}

fn selected_message_call_ids(messages: &[Message]) -> HashSet<String> {
    let calls = messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    let recent_start = calls.len().saturating_sub(RECENT_CHECKPOINT_TOOL_CALLS);
    let first_mutation = calls
        .iter()
        .find(|call| is_mutating_tool_call(&call.name, &call.arguments))
        .map(|call| call.id.as_str());
    calls
        .iter()
        .enumerate()
        .filter(|(index, call)| *index >= recent_start || Some(call.id.as_str()) == first_mutation)
        .map(|(_, call)| call.id.clone())
        .collect()
}

fn compact_responses_input(input: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let Some(boundary) = input.iter().position(raw_item_is_mutation) else {
        return input.to_vec();
    };
    let calls = input[boundary..]
        .iter()
        .filter(|item| {
            item.get("type").and_then(serde_json::Value::as_str) == Some("function_call")
        })
        .collect::<Vec<_>>();
    let recent_start = calls.len().saturating_sub(RECENT_CHECKPOINT_TOOL_CALLS);
    let first_mutation = calls
        .iter()
        .find(|item| raw_item_is_mutation(item))
        .and_then(|item| item.get("call_id"))
        .and_then(serde_json::Value::as_str);
    let selected_call_ids = calls
        .iter()
        .enumerate()
        .filter(|(index, item)| {
            *index >= recent_start
                || item.get("call_id").and_then(serde_json::Value::as_str) == first_mutation
        })
        .filter_map(|(_, item)| item.get("call_id").and_then(serde_json::Value::as_str))
        .collect::<HashSet<_>>();
    input[..boundary]
        .iter()
        .cloned()
        .chain(
            input[boundary..]
                .iter()
                .filter(|item| {
                    let kind = item.get("type").and_then(serde_json::Value::as_str);
                    matches!(kind, Some("function_call") | Some("function_call_output"))
                        && item
                            .get("call_id")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|call_id| selected_call_ids.contains(call_id))
                })
                .cloned(),
        )
        .collect()
}

fn raw_item_is_mutation(item: &serde_json::Value) -> bool {
    if item.get("type").and_then(serde_json::Value::as_str) != Some("function_call") {
        return false;
    }
    let Some(name) = item.get("name").and_then(serde_json::Value::as_str) else {
        return false;
    };
    is_mutating_tool_call(
        name,
        item.get("arguments").unwrap_or(&serde_json::Value::Null),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, InstructionBlock, LlmRequest, Message, Metadata, Request, ResponseOutput,
        Role, ToolCall, ToolResult,
    };

    use super::*;
    use crate::core::testing::{reply, test_drive};

    fn algorithm() -> Arc<dyn Algorithm> {
        Arc::new(
            PlanExecute::new(
                ModelId::from("model/capable"),
                ModelId::from("model/efficient"),
                PlanExecuteConfig::default(),
            )
            .expect("default config should be valid"),
        )
    }

    fn checkpoint_algorithm(
        interval: u32,
        max_checkpoints: u32,
        max_turns: u32,
        max_turns_total: u32,
    ) -> Arc<dyn Algorithm> {
        Arc::new(
            PlanExecute::new(
                ModelId::from("model/capable"),
                ModelId::from("model/efficient"),
                PlanExecuteConfig {
                    checkpoint_interval_turns: interval,
                    max_checkpoints,
                    max_checkpoint_turns: max_turns,
                    max_checkpoint_turns_total: max_turns_total,
                    ..PlanExecuteConfig::default()
                },
            )
            .expect("checkpoint config should be valid"),
        )
    }

    fn request(messages: Vec<Message>, session_id: Option<&str>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/plan-execute".to_string()),
                messages,
                ..LlmRequest::default()
            },
            metadata: session_id.map(|session_id| Metadata {
                session_id: Some(session_id.to_string()),
                ..Metadata::default()
            }),
            ..Request::default()
        }
    }

    fn tool_call(name: &str, arguments: serde_json::Value) -> Message {
        tool_call_with_id("call-1", name, arguments)
    }

    fn tool_call_with_id(id: &str, name: &str, arguments: serde_json::Value) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })],
        }
    }

    fn tool_result(call_id: &str, output: &str) -> Message {
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: call_id.to_string(),
                content: vec![ContentBlock::Text {
                    text: output.to_string(),
                }],
                is_error: Some(false),
            })],
        }
    }

    fn model_tool_response(name: &str, arguments: serde_json::Value, text: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Text {
                            text: text.to_string(),
                        },
                        ContentBlock::ToolCall(ToolCall {
                            id: "checkpoint-call".to_string(),
                            name: name.to_string(),
                            arguments,
                        }),
                    ],
                    stop_reason: None,
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    async fn drive_and_capture_calls(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
        capable_response: Response,
    ) -> (ModelId, Response, Vec<(ModelId, Request)>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let capable_response = Arc::new(Mutex::new(Some(capable_response)));
        let (selected, response) = test_drive(
            algorithm,
            request,
            move |target: ModelId, request: Request| {
                let captured = Arc::clone(&captured);
                let capable_response = Arc::clone(&capable_response);
                async move {
                    captured
                        .lock()
                        .expect("capture lock should be available")
                        .push((target.clone(), request));
                    if target == "model/capable" {
                        Ok(capable_response
                            .lock()
                            .expect("capable response lock should be available")
                            .take()
                            .expect("capable model should be called once"))
                    } else {
                        Ok(reply("efficient"))
                    }
                }
            },
        )
        .await
        .expect("routing should succeed");
        let calls = calls
            .lock()
            .expect("capture lock should be available")
            .clone();
        (selected, response, calls)
    }

    async fn route_and_capture(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
    ) -> (ModelId, Request) {
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let (selected, _) = test_drive(algorithm, request, move |_target, request| {
            let capture = Arc::clone(&capture);
            async move {
                *capture.lock().expect("capture lock should be available") = Some(request);
                Ok(reply("ok"))
            }
        })
        .await
        .expect("routing should succeed");
        let request = captured
            .lock()
            .expect("capture lock should be available")
            .take()
            .expect("answer request should be captured");
        (selected, request)
    }

    #[tokio::test]
    async fn initial_turn_uses_capable_model_with_planning_prefix() {
        let messages = vec![Message::text(Role::User, "fix the parser")];
        let (selected, routed) =
            route_and_capture(algorithm(), request(messages.clone(), Some("task-1"))).await;

        assert_eq!(selected, "model/capable");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
        assert_eq!(routed.llm_request.instructions[0].role, Role::System);
        assert_eq!(
            routed.llm_request.instructions[0].content,
            vec![ContentBlock::Text {
                text: DEFAULT_PLANNING_PROMPT.trim().to_string()
            }]
        );
    }

    #[tokio::test]
    async fn read_only_tool_calls_remain_in_planning() {
        let messages = vec![
            Message::text(Role::User, "fix the parser"),
            tool_call("exec_command", json!({"cmd": "rg parser crates"})),
        ];

        let (selected, routed) =
            route_and_capture(algorithm(), request(messages.clone(), None)).await;

        assert_eq!(selected, "model/capable");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
    }

    #[tokio::test]
    async fn first_edit_switches_to_efficient_and_keeps_the_trajectory() {
        let messages = vec![
            Message::text(Role::User, "fix the parser"),
            Message::text(Role::Assistant, "I will update the parser now."),
            tool_call("apply_patch", json!({"patch": "*** Begin Patch"})),
        ];
        let mut input = request(messages.clone(), Some("task-2"));
        input.llm_request.instructions.push(InstructionBlock {
            role: Role::Developer,
            content: vec![ContentBlock::Text {
                text: "keep the public API stable".to_string(),
            }],
        });

        let (selected, routed) = route_and_capture(algorithm(), input).await;

        assert_eq!(selected, "model/efficient");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
        assert_eq!(routed.llm_request.instructions[0].role, Role::Developer);
    }

    #[tokio::test]
    async fn shell_file_write_switches_to_execution() {
        let messages = vec![tool_call(
            "exec_command",
            json!({"cmd": "python -c 'from pathlib import Path; Path(\"x\").write_text(\"y\")'"}),
        )];

        let (selected, routed) = route_and_capture(algorithm(), request(messages, None)).await;

        assert_eq!(selected, "model/efficient");
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn shell_redirection_switches_to_execution() {
        let messages = vec![tool_call(
            "exec_command",
            json!({"cmd": "printf 'completed\\n' > task.txt"}),
        )];

        let (selected, routed) = route_and_capture(algorithm(), request(messages, None)).await;

        assert_eq!(selected, "model/efficient");
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn execution_latches_by_session_after_history_compaction() {
        let algorithm = algorithm();
        let edit = request(
            vec![tool_call("write_file", json!({"path": "src/lib.rs"}))],
            Some("task-3"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), edit).await;
        assert_eq!(selected, "model/efficient");

        let compacted = request(
            vec![Message::text(
                Role::User,
                "Continue from the compacted summary",
            )],
            Some("task-3"),
        );
        let (selected, routed) = route_and_capture(algorithm, compacted).await;

        assert_eq!(selected, "model/efficient");
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn final_request_uses_then_releases_the_session_latch() {
        let algorithm = algorithm();
        let edit = request(
            vec![tool_call("write_file", json!({"path": "src/lib.rs"}))],
            Some("task-4"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), edit).await;
        assert_eq!(selected, "model/efficient");

        let mut final_request = request(vec![Message::text(Role::User, "Finish")], Some("task-4"));
        final_request
            .metadata
            .as_mut()
            .expect("session metadata should exist")
            .session_final = Some(true);
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), final_request).await;
        assert_eq!(selected, "model/efficient");

        let reused = request(vec![Message::text(Role::User, "New task")], Some("task-4"));
        let (selected, _) = route_and_capture(algorithm, reused).await;
        assert_eq!(selected, "model/capable");
    }

    #[tokio::test]
    async fn checkpoint_allows_read_only_probe_then_executes_one_capable_edit() {
        let algorithm = checkpoint_algorithm(2, 8, 4, 16);
        let seed = request(
            vec![tool_call_with_id(
                "initial-edit",
                "apply_patch",
                json!({"patch": "initial"}),
            )],
            Some("checkpoint-task"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), seed).await;
        assert_eq!(selected, "model/efficient");

        let progress = request(
            vec![Message::text(Role::User, "implementation progress")],
            Some("checkpoint-task"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), progress).await;
        assert_eq!(selected, "model/efficient");

        let mut checkpoint = request(
            vec![
                Message::text(Role::User, "inspect current state"),
                tool_call_with_id("initial-edit", "apply_patch", json!({"patch": "initial"})),
                tool_result("initial-edit", "Done"),
            ],
            Some("checkpoint-task"),
        );
        checkpoint.llm_request.preservation.requests.insert(
            FormatId::known(WireFormat::OpenAiResponses),
            json!({
                "instructions": "client instructions",
                "input": [
                    {"type": "message", "role": "user", "content": "inspect current state"},
                    {"type": "function_call", "call_id": "initial-edit", "name": "apply_patch", "arguments": "{\"patch\":\"initial\"}"},
                    {"type": "function_call_output", "call_id": "initial-edit", "output": "Done"}
                ],
                "stream": true
            }),
        );
        let (selected, response, calls) = drive_and_capture_calls(
            Arc::clone(&algorithm),
            checkpoint,
            model_tool_response(
                "exec_command",
                json!({"cmd": "git diff --stat"}),
                "I will inspect the current patch.",
            ),
        )
        .await;
        assert_eq!(selected, "model/capable");
        assert!(matches!(response.llm_response, LlmResponse::Agg(_)));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "model/capable");
        assert!(calls[0].1.llm_request.stream);
        assert!(calls[0].1.llm_request.preservation.requests.is_empty());
        assert_eq!(calls[0].1.llm_request.instructions.len(), 1);
        assert_eq!(
            calls[0].1.llm_request.instructions[0].content,
            vec![ContentBlock::Text {
                text: DEFAULT_PLANNING_PROMPT.trim().to_string()
            }]
        );

        let mutation = request(
            vec![
                Message::text(Role::User, "inspect current state"),
                tool_call_with_id("initial-edit", "apply_patch", json!({"patch": "initial"})),
                tool_result("initial-edit", "Done"),
                tool_call_with_id(
                    "checkpoint-read",
                    "exec_command",
                    json!({"cmd": "git diff --stat"}),
                ),
                tool_result("checkpoint-read", "src/lib.rs | 2 ++"),
            ],
            Some("checkpoint-task"),
        );
        let (selected, response, calls) = drive_and_capture_calls(
            Arc::clone(&algorithm),
            mutation,
            model_tool_response(
                "apply_patch",
                json!({"patch": "*** Begin Patch\n*** Update File: src/lib.rs"}),
                "The implementation needs one correction.",
            ),
        )
        .await;
        assert_eq!(selected, "model/capable");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "model/capable");
        let LlmResponse::Agg(aggregate) = response.llm_response else {
            panic!("checkpoint edit should be buffered")
        };
        assert_eq!(mutating_calls(&aggregate).len(), 1);

        let edit_executed = request(
            vec![
                Message::text(Role::Assistant, "The implementation needs one correction."),
                tool_call_with_id("initial-edit", "apply_patch", json!({"patch": "initial"})),
                tool_result("initial-edit", "Done"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "checkpoint-call".to_string(),
                        name: "apply_patch".to_string(),
                        arguments: json!({"patch": "*** Begin Patch\n*** Update File: src/lib.rs"}),
                    })],
                },
                tool_result("checkpoint-call", "Done"),
            ],
            Some("checkpoint-task"),
        );
        let (selected, routed) = route_and_capture(algorithm, edit_executed.clone()).await;
        assert_eq!(selected, "model/efficient");
        assert_eq!(
            routed.llm_request.messages,
            edit_executed.llm_request.messages
        );
    }

    #[tokio::test]
    async fn checkpoint_turn_cap_forces_a_tool_free_handoff() {
        let algorithm = checkpoint_algorithm(1, 8, 1, 16);
        let seed = request(
            vec![tool_call("apply_patch", json!({"patch": "initial"}))],
            Some("bounded-task"),
        );
        route_and_capture(Arc::clone(&algorithm), seed).await;

        let checkpoint = request(
            vec![Message::text(Role::User, "continue")],
            Some("bounded-task"),
        );
        let (selected, _, calls) = drive_and_capture_calls(
            algorithm,
            checkpoint,
            Response {
                llm_response: LlmResponse::Agg(switchyard_protocol::text_response(
                    None,
                    "The current direction is sound.",
                )),
                metadata: None,
            },
        )
        .await;
        assert_eq!(selected, "model/efficient");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "model/capable");
        assert_eq!(calls[0].1.llm_request.tool_choice, Some(ToolChoice::None));
    }

    #[tokio::test]
    async fn total_checkpoint_cap_disables_later_checkpoints() {
        let algorithm = checkpoint_algorithm(1, 8, 1, 1);
        let seed = request(
            vec![tool_call("apply_patch", json!({"patch": "initial"}))],
            Some("total-cap-task"),
        );
        route_and_capture(Arc::clone(&algorithm), seed).await;

        let checkpoint = request(
            vec![Message::text(Role::User, "continue")],
            Some("total-cap-task"),
        );
        drive_and_capture_calls(
            Arc::clone(&algorithm),
            checkpoint,
            Response {
                llm_response: LlmResponse::Agg(switchyard_protocol::text_response(None, "done")),
                metadata: None,
            },
        )
        .await;

        let progress = request(
            vec![Message::text(Role::User, "more progress")],
            Some("total-cap-task"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), progress).await;
        assert_eq!(selected, "model/efficient");
        let next = request(
            vec![Message::text(Role::User, "continue again")],
            Some("total-cap-task"),
        );
        let (selected, _) = route_and_capture(algorithm, next).await;
        assert_eq!(selected, "model/efficient");
    }

    #[test]
    fn response_compaction_keeps_prefix_first_mutation_and_recent_evidence() {
        let mut input = vec![
            json!({"type": "message", "role": "user", "content": "task"}),
            json!({"type": "function_call", "call_id": "plan-read", "name": "exec_command", "arguments": "{\"cmd\":\"rg parser\"}"}),
            json!({"type": "function_call_output", "call_id": "plan-read", "output": "found"}),
            json!({"type": "function_call", "call_id": "first-edit", "name": "apply_patch", "arguments": "{\"patch\":\"first\"}"}),
            json!({"type": "function_call_output", "call_id": "first-edit", "output": "done"}),
        ];
        for index in 0..8 {
            input.push(json!({"type": "function_call", "call_id": format!("read-{index}"), "name": "exec_command", "arguments": "{\"cmd\":\"git diff\"}"}));
            input.push(json!({"type": "function_call_output", "call_id": format!("read-{index}"), "output": format!("result-{index}")}));
        }

        let compacted = compact_responses_input(&input);
        let encoded = serde_json::to_string(&compacted).expect("compacted input should encode");
        assert!(encoded.contains("plan-read"));
        assert!(encoded.contains("first-edit"));
        assert!(!encoded.contains("read-0"));
        assert!(!encoded.contains("read-1"));
        assert!(encoded.contains("read-2"));
        assert!(encoded.contains("read-7"));
    }

    #[test]
    fn empty_planning_prompt_is_rejected() {
        let result = PlanExecute::new(
            ModelId::from("model/capable"),
            ModelId::from("model/efficient"),
            PlanExecuteConfig {
                planning_prompt: "  ".to_string(),
                ..PlanExecuteConfig::default()
            },
        );

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
    }
}
