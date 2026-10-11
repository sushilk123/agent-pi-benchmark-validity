"""
AgentDojo LLM backend that routes generation through Amazon Bedrock `converse`,
so AgentDojo runs under the SAME uniform decoding config as our other benchmarks
(provider-default, no explicit temperature — valid for every vendor incl. Claude 5).

We subclass AgentDojo's PromptingLLM (text-prompted tool use) and override only the
raw completion: its tool-format prompting and output parsing are reused unchanged.

NOT FOR DEPLOYMENT — local research script; creds via boto3.
Status: import/syntax-checked; end-to-end run pending (see run_agentdojo.py).
"""
import sys
import sys  # noqa: E401
sys.path.insert(0, __import__('os').path.dirname(__file__))
from paths import ROOT, INJECAGENT_DIR, BIPIA_DIR, RESULTS_DIR, LOG_DIR  # noqa: E402,F401
from types import SimpleNamespace
import boto3
from openai.types.chat import ChatCompletionMessage

from agentdojo.agent_pipeline.llms.prompting_llm import (
    PromptingLLM, _message_to_together, InvalidModelOutputError,
)
from agentdojo.ast_utils import ASTParsingError
from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage, ChatMessage, ChatSystemMessage, ChatUserMessage,
    get_text_content_as_str, text_content_block_from_string,
)
from collections.abc import Sequence


class BedrockPromptingLLM(PromptingLLM):
    """PromptingLLM whose completions come from Bedrock converse (provider-default)."""

    def __init__(self, model_id: str, name: str, region: str = "us-west-2"):
        self.model = model_id
        self.name = name
        self.temperature = None            # unused; converse uses provider default
        self.client = None                 # OpenAI client not used
        self._brt = boto3.client("bedrock-runtime", region_name=region)

    # --- the only substantive override: produce a completion via converse ---
    def _complete(self, openai_messages):
        system_text = "\n\n".join(
            m["content"] for m in openai_messages
            if m.get("role") in ("system", "developer") and isinstance(m.get("content"), str))
        convo = [m for m in openai_messages if m.get("role") not in ("system", "developer")]
        user_text = "\n\n".join(f'{m.get("role")}: {m.get("content","")}' for m in convo) or "(begin)"
        try:
            r = self._brt.converse(
                modelId=self.model,
                system=[{"text": system_text or "You are a helpful assistant."}],
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                inferenceConfig={"maxTokens": 1024})
            text = " ".join(p.get("text", "") for p in r["output"]["message"]["content"])
        except Exception as e:
            print(f"    converse err {self.model}: {str(e)[:80]}", file=sys.stderr)
            text = ""
        msg = ChatCompletionMessage(role="assistant", content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def query(self, query, runtime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        adapted = [self._tool_message_to_user_message(m) if m["role"] == "tool" else m
                   for m in messages]
        openai_messages = [_message_to_together(m, self.model) for m in adapted]
        system_message, _ = self._get_system_message(messages)
        system_message = self._make_tools_prompt(system_message, list(runtime.functions.values()))
        if system_message is not None:
            openai_messages = [{"role": "system",
                                "content": get_text_content_as_str(system_message["content"] or [])},
                               *openai_messages]
        completion = self._complete(openai_messages)
        output = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(completion.choices[0].message.content or "")],
            tool_calls=[])
        if len(runtime.functions) == 0 or "<function-call>" not in get_text_content_as_str(output["content"] or []):
            return query, runtime, env, [*messages, output], extra_args
        for _ in range(self._MAX_ATTEMPTS):
            try:
                output = self._parse_model_output(completion.choices[0].message)
                break
            except (InvalidModelOutputError, ASTParsingError, IndexError,
                    SyntaxError, ValueError) as e:
                # Weak/non-Claude models often emit malformed tool calls. Retry with
                # feedback; if it never parses, `output` keeps the plain-text message
                # (built above) so the rollout ends gracefully instead of crashing.
                err = ChatUserMessage(role="user",
                                      content=[text_content_block_from_string(f"Invalid function calling output: {e!s}")])
                try:
                    completion = self._complete([*openai_messages, _message_to_openai(err, self.model)])
                except Exception:
                    break
        return query, runtime, env, [*messages, output], extra_args
