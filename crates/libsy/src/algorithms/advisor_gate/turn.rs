// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One buffered executor turn: consuming it to completion, inspecting it,
//! and replaying it to the client verbatim.

use futures::StreamExt;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmClientError, LlmResponse, LlmResponseChunk,
    LlmResponseStreamEvent, Metadata, Response, ResponseAccumulator, StopReason, WireFormat,
};

use crate::{LibsyError, Result};

// ── Turn buffering and replay ───────────────────────────────────────────────

/// One fully generated executor turn held while the gate decides.
pub(super) struct GatedTurn {
    /// Buffered provider events for streamed turns, preservation included, so
    /// replay re-emits them verbatim (signed thinking and provider extensions
    /// survive; folding to an aggregate and re-synthesizing would drop them).
    pub(super) events: Option<Vec<LlmResponseStreamEvent>>,
    /// Folded view for detection, the review tail, the REDO echo, and
    /// discarded-turn usage. For buffered turns this is the original
    /// response, its own preservation intact.
    pub(super) agg: AggLlmResponse,
    pub(super) metadata: Option<Metadata>,
}

impl GatedTurn {
    /// Releases the turn to the client: streamed turns replay their buffered
    /// events verbatim, buffered turns return the original aggregate.
    pub(super) fn into_response(self) -> Response {
        let llm_response = match self.events {
            Some(events) => {
                LlmResponse::Stream(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
            }
            None => LlmResponse::Agg(self.agg),
        };
        Response {
            llm_response,
            metadata: self.metadata,
        }
    }
}

/// Consumes the executor response to completion. Mid-stream failures — item
/// errors and in-band error chunks — become typed client-call errors exactly
/// as [`LlmResponse::into_agg`] maps them; the client saw nothing yet, so the
/// turn fails whole.
pub(super) async fn buffer_turn(executor: &str, response: Response) -> Result<GatedTurn> {
    let metadata = response.metadata;
    match response.llm_response {
        LlmResponse::Agg(agg) => Ok(GatedTurn {
            events: None,
            agg,
            metadata,
        }),
        LlmResponse::Stream(mut stream) => {
            let mut events = Vec::new();
            let mut accumulator = ResponseAccumulator::new();
            while let Some(item) = stream.next().await {
                let event =
                    item.map_err(|source| LibsyError::client_call(executor.to_string(), source))?;
                for chunk in event.normalized() {
                    let failure = match chunk {
                        LlmResponseChunk::DecodeError { message } => {
                            Some(LlmClientError::ResponseTranslation(message.clone()))
                        }
                        LlmResponseChunk::StreamError { message } => {
                            Some(LlmClientError::UpstreamHttp {
                                status: http::StatusCode::BAD_GATEWAY,
                                body: message.clone(),
                            })
                        }
                        chunk => {
                            accumulator.push(chunk.clone());
                            None
                        }
                    };
                    if let Some(source) = failure {
                        return Err(LibsyError::client_call(executor.to_string(), source));
                    }
                }
                events.push(event);
            }
            let mut agg = accumulator.finish();
            if let Some(response) = events.iter().rev().find_map(|event| {
                let preserved = event.preservation()?;
                (preserved.source().as_str() == WireFormat::OpenAiResponses.as_str())
                    .then(|| preserved.raw().get("response"))
                    .flatten()
                    .filter(|response| response.get("output").is_some())
                    .cloned()
            }) {
                agg.preservation
                    .responses
                    .insert(WireFormat::OpenAiResponses.into(), response);
            }
            Ok(GatedTurn {
                events: Some(events),
                agg,
                metadata,
            })
        }
    }
}

// ── Detection over the folded turn ──────────────────────────────────────────

/// Whether the turn carries tool use on either signal: a `ToolUse` stop
/// reason, or any tool-call block (some OSS servers mislabel tool-call turns
/// as an ordinary stop, so block presence wins).
pub(super) fn has_tool_use(agg: &AggLlmResponse) -> bool {
    agg.outputs.iter().any(|output| {
        output.stop_reason == Some(StopReason::ToolUse)
            || output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    })
}

/// Whether a preserved Responses payload marks an assistant message as final.
pub(super) fn is_final_answer(agg: &AggLlmResponse) -> bool {
    let responses = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    agg.preservation
        .responses
        .get(&responses)
        .and_then(|response| response.get("output"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.get("type").and_then(serde_json::Value::as_str) == Some("message")
                    && item.get("phase").and_then(serde_json::Value::as_str) == Some("final_answer")
            })
        })
}

/// The turn's visible text: all text blocks joined; empty means none.
pub(super) fn visible_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// The turn's internal reasoning, the review evidence of last resort.
pub(super) fn reasoning_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmResponseStreamEvent, Response};

    fn final_answer_response() -> serde_json::Value {
        serde_json::json!({
            "id": "resp-final",
            "output": [{
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "finished"}]
            }]
        })
    }

    #[test]
    fn buffered_responses_final_answer_is_detected() {
        let mut agg = AggLlmResponse::default();
        agg.preservation
            .responses
            .insert(WireFormat::OpenAiResponses.into(), final_answer_response());
        assert!(is_final_answer(&agg));
    }

    #[tokio::test]
    async fn streamed_responses_final_answer_is_detected() {
        let event = LlmResponseStreamEvent::preserved(
            WireFormat::OpenAiResponses,
            serde_json::json!({
                "type": "response.completed",
                "response": final_answer_response()
            }),
            vec![
                LlmResponseChunk::MessageStart {
                    id: Some("resp-final".to_string()),
                    model: Some("executor".to_string()),
                },
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "finished".to_string(),
                },
                LlmResponseChunk::MessageStop { reason: None },
            ],
        );
        let turn = buffer_turn(
            "executor",
            Response {
                llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter([Ok(event)]))),
                metadata: None,
            },
        )
        .await
        .expect("stream buffers");
        assert!(is_final_answer(&turn.agg));
    }
}
