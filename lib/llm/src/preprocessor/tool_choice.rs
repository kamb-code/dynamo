// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::get_json_schema_from_tools;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

/// Which generation constraint `apply_tool_choice_guided_decoding` actually installed.
///
/// The postprocessor cannot re-derive this from `tool_choice` alone. A Kimi K3
/// `required` request returns [`Self::None`] because it is served by a prompt-level
/// XTML instruction and no JSON schema is installed, yet its `tool_choice` is
/// identical to a request that IS guided-JSON constrained. Routing on `tool_choice`
/// would stream that XTML into `function.arguments`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GuidedToolConstraint {
    /// No tool-choice constraint installed by this function.
    None,
    /// Generation pinned to the family's NATIVE markup grammar, not JSON.
    StructuralTag,
    /// Output is the named tool's ARGUMENT object alone; the name is known from the request.
    GuidedJsonNamed { tool_name: String },
    /// Output is a JSON array of `{name, parameters}` objects.
    GuidedJsonRequired,
}

impl GuidedToolConstraint {
    /// Whether a JSON grammar was INSTALLED for this request's tool calls.
    ///
    /// `true` means the payload's shape is known ahead of time, so the jail may
    /// release calls as they arrive. It does NOT mean the backend will honour the
    /// grammar - the jail keeps its native fallback for backends that do not.
    pub(crate) fn installs_guided_json(&self) -> bool {
        matches!(
            self,
            Self::GuidedJsonNamed { .. } | Self::GuidedJsonRequired
        )
    }

    /// True when generation was pinned to native markup via a structural tag.
    ///
    /// Preserves the boolean this function used to return, so existing callers
    /// keep their exact previous behaviour.
    pub(crate) fn uses_structural_tag(&self) -> bool {
        matches!(self, Self::StructuralTag)
    }
}

impl OpenAIPreprocessor {
    /// Whether this request permits model output to be interpreted as tool calls.
    ///
    /// A configured parser describes the model's wire format; it does not grant
    /// every request permission to return tool calls. Permission depends only on
    /// whether the request supplies tools and whether `tool_choice` forbids them.
    /// Assistant-output constraints apply to assistant content and do not revoke
    /// an `auto` request's ability to choose a tool call.
    pub(crate) fn tool_call_parsing_enabled(request: &NvCreateChatCompletionRequest) -> bool {
        if request.inner.tools.as_ref().is_none_or(Vec::is_empty) {
            return false;
        }

        match request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto)
        {
            ChatCompletionToolChoiceOption::None => false,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                true
            }
            ChatCompletionToolChoiceOption::Auto => true,
        }
    }

    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Supported K2 forced requests and named K3 requests
    /// intrinsically use their native structural tags because generic JSON cannot
    /// represent their tool calls. Other forced choices fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied, except K3
    /// required requests, which stay on the prompt-level XTML path.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<GuidedToolConstraint, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = request.inner.tools.as_deref().unwrap_or(&[]);
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(GuidedToolConstraint::None);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        let uses_kimi_k3_parser = uses_kimi_k3_parser(
            self.tool_call_parser.as_deref(),
            self.runtime_config.reasoning_parser.as_deref(),
        );
        if is_forced_tool_choice && uses_kimi_k3_parser {
            if matches!(tool_choice, ChatCompletionToolChoiceOption::Named(_)) {
                return Err(invalid_argument(
                    "named tool choice for Kimi K3 requires --dyn-tool-call-parser kimi_k3 \
                     with XTML structural-tag support",
                ));
            }

            // K3's prompt-level required instruction produces an XTML `tools`
            // channel. Generic JSON guided decoding would constrain the wrong
            // wire format and prevent the Rust K3 parser from seeing it. No JSON
            // schema is installed, so this is NOT a guided-JSON request.
            return Ok(GuidedToolConstraint::None);
        }

        match get_json_schema_from_tools(
            Some(tool_choice),
            Some(tools),
            request.inner.parallel_tool_calls,
        ) {
            Ok(Some(schema)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                gd.json = Some(schema);

                // Report the shape that was installed, not the tool_choice that
                // asked for it. Only these two arms produce a JSON schema, so only
                // they may be streamed as guided JSON.
                return Ok(installed_json_constraint(tool_choice));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(GuidedToolConstraint::None)
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided.
///
/// This is the SHARED owner of that decision. `apply_tool_choice_guided_decoding` calls
/// it to report what it installed, and the postprocessor calls it to route the stream.
/// Routing must never re-derive the answer from `tool_choice` alone: a Kimi K3 forced
/// request looks identical there but installs no JSON schema, so its XTML would be
/// streamed into `function.arguments`.
pub(crate) fn guided_tool_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> GuidedToolConstraint {
    if uses_structural_tag {
        return GuidedToolConstraint::StructuralTag;
    }
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    let is_forced_tool_choice = matches!(
        tool_choice,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
    );
    // Only a forced choice installs a tool-level JSON schema. Auto/None leave any
    // JSON constraint to `response_format`, which governs assistant CONTENT.
    if !is_forced_tool_choice {
        return GuidedToolConstraint::None;
    }
    // Forced + explicit guided decoding is rejected at request time, so reaching here
    // with both means the request never ran.
    if has_explicit_guided_decoding(request) {
        return GuidedToolConstraint::None;
    }
    // K3 forced requests are served by a prompt-level XTML instruction; no JSON schema.
    if uses_kimi_k3_parser(tool_call_parser, reasoning_parser) {
        return GuidedToolConstraint::None;
    }
    installed_json_constraint(tool_choice)
}

/// True when either configured parser is Kimi K3.
///
/// K3 forced requests are served by a prompt-level XTML instruction, so they must
/// NOT be reported as guided JSON even though their `tool_choice` is forced.
fn uses_kimi_k3_parser(tool_call_parser: Option<&str>, reasoning_parser: Option<&str>) -> bool {
    let is_k3 = |parser: &str| matches!(parser, "kimi_k3" | "kimi-k3");
    tool_call_parser.is_some_and(is_k3) || reasoning_parser.is_some_and(is_k3)
}

/// Map a forced `tool_choice` onto the JSON shape its schema constrains output to.
///
/// Only reachable once `get_json_schema_from_tools` actually produced a schema, and
/// that function returns `None` for `Auto`/`None`, so those arms are unreachable in
/// practice and report [`GuidedToolConstraint::None`] rather than guessing.
fn installed_json_constraint(tool_choice: &ChatCompletionToolChoiceOption) -> GuidedToolConstraint {
    match tool_choice {
        ChatCompletionToolChoiceOption::Named(named) => GuidedToolConstraint::GuidedJsonNamed {
            tool_name: named.function.name.clone(),
        },
        ChatCompletionToolChoiceOption::Required => GuidedToolConstraint::GuidedJsonRequired,
        ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None => {
            GuidedToolConstraint::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{ChatCompletionNamedToolChoice, FunctionName};
    use serde_json::{Value, json};

    fn request(extra: Value) -> NvCreateChatCompletionRequest {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}]
        });
        value
            .as_object_mut()
            .expect("base request is an object")
            .extend(extra.as_object().expect("extra is an object").clone());
        serde_json::from_value(value).expect("request must deserialize")
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }])
    }

    #[test]
    fn tool_call_parsing_requires_tools() {
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({})
        )));
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({"tool_choice": "required"})
        )));
    }

    #[test]
    fn tool_call_parsing_honors_each_tool_choice() {
        for (tool_choice, expected) in [
            (json!(null), true),
            (json!("none"), false),
            (json!("auto"), true),
            (json!("required"), true),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                true,
            ),
        ] {
            let mut extra = json!({"tools": tools()});
            if !tool_choice.is_null() {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn assistant_constraints_do_not_revoke_auto_tool_permission() {
        let constraints = [
            json!({"response_format": {"type": "json_object"}}),
            json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "event",
                        "schema": {"type": "object"}
                    }
                }
            }),
            json!({"guided_json": {"type": "object"}}),
            json!({"guided_regex": "[a-z]+"}),
            json!({"guided_choice": ["a", "b"]}),
            json!({"guided_grammar": "root ::= 'a'"}),
        ];

        for constraint in constraints {
            let mut auto = json!({"tools": tools(), "tool_choice": "auto"});
            auto.as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                auto
            )));

            let mut required = json!({"tools": tools(), "tool_choice": "required"});
            required
                .as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                required
            )));
        }
    }

    #[test]
    fn text_response_format_does_not_disable_auto_tool_parsing() {
        assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({
                "tools": tools(),
                "tool_choice": "auto",
                "response_format": {"type": "text"}
            })
        )));
    }

    fn named(name: &str) -> ChatCompletionToolChoiceOption {
        ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
            r#type: dynamo_protocols::types::ChatCompletionToolType::Function,
            function: FunctionName {
                name: name.to_string(),
            },
        })
    }

    #[test]
    fn only_structural_tag_reports_a_structural_tag() {
        assert!(GuidedToolConstraint::StructuralTag.uses_structural_tag());
        assert!(!GuidedToolConstraint::None.uses_structural_tag());
        assert!(!GuidedToolConstraint::GuidedJsonRequired.uses_structural_tag());
        assert!(
            !GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
            .uses_structural_tag()
        );
    }

    #[test]
    fn required_reports_the_array_shape() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Required),
            GuidedToolConstraint::GuidedJsonRequired
        );
    }

    #[test]
    fn named_carries_the_tool_name_from_the_request() {
        assert_eq!(
            installed_json_constraint(&named("get_weather")),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
        );
    }

    #[test]
    fn unforced_choices_install_no_tool_constraint() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Auto),
            GuidedToolConstraint::None
        );
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::None),
            GuidedToolConstraint::None
        );
    }

    #[test]
    fn kimi_k3_is_detected_from_either_parser_slot() {
        assert!(uses_kimi_k3_parser(Some("kimi_k3"), None));
        assert!(uses_kimi_k3_parser(Some("kimi-k3"), None));
        assert!(uses_kimi_k3_parser(None, Some("kimi_k3")));
        assert!(uses_kimi_k3_parser(None, Some("kimi-k3")));
    }

    #[test]
    fn non_k3_parsers_are_not_mistaken_for_k3() {
        assert!(!uses_kimi_k3_parser(None, None));
        assert!(!uses_kimi_k3_parser(Some("kimi_k2"), Some("qwen3")));
        assert!(!uses_kimi_k3_parser(Some("qwen3_coder"), None));
    }
}
