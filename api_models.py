import openai
from openai import OpenAI
from tqdm import tqdm
from typing import List, Optional
import time
import re
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_openrouter_deepseek_model_name(model_name: str) -> str:
    """
    Normalize shorthand DeepSeek model names to OpenRouter IDs.
    Example: deepseek-v3.2 -> deepseek/deepseek-v3.2
    """
    if not isinstance(model_name, str):
        return model_name

    stripped = model_name.strip()
    lowered = stripped.lower()

    if lowered.startswith("deepseek/"):
        return stripped
    if lowered.startswith("deepseek-"):
        return f"deepseek/{stripped}"
    return stripped


def _normalize_openrouter_grok_model_name(model_name: str) -> str:
    """
    Normalize shorthand Grok model names to OpenRouter IDs.
    Example: grok-4.1-fast -> x-ai/grok-4.1-fast
    """
    if not isinstance(model_name, str):
        return model_name

    stripped = model_name.strip()
    lowered = stripped.lower()
    if lowered.startswith("x-ai/"):
        return stripped
    if lowered.startswith("grok-"):
        return f"x-ai/{stripped}"
    return stripped


def _normalize_openrouter_gemini_model_name(model_name: str) -> str:
    """
    Normalize shorthand Gemini model names to OpenRouter IDs.
    Example: gemini_3_flash_preview -> google/gemini-3-flash-preview
    """
    if not isinstance(model_name, str):
        return model_name

    stripped = model_name.strip()
    lowered = stripped.lower()

    if lowered.startswith("google/"):
        return stripped

    alias_map = {
        "gemini_3_flash_preview": "google/gemini-3-flash-preview",
        "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    }
    if lowered in alias_map:
        return alias_map[lowered]

    if lowered.startswith("gemini_"):
        return f"google/{lowered.replace('_', '-')}"
    if lowered.startswith("gemini-"):
        return f"google/{lowered}"
    return stripped


def _normalize_openrouter_claude_model_name(model_name: str) -> str:
    """
    Normalize shorthand Claude model names to OpenRouter IDs.
    Example: claude-sonnet-4.5 -> anthropic/claude-sonnet-4.5
    """
    if not isinstance(model_name, str):
        return model_name

    stripped = model_name.strip()
    lowered = stripped.lower()

    if lowered.startswith("anthropic/"):
        return stripped
    if lowered.startswith("claude-"):
        return f"anthropic/{stripped}"
    return stripped


def _resolve_api_key(token: Optional[str]) -> Optional[str]:
    token_is_placeholder = False
    if isinstance(token, str):
        t = token.strip()
        if t and not (t.startswith("<") and t.endswith(">")):
            return t
        if t.startswith("<") and t.endswith(">"):
            token_is_placeholder = True

    env_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if env_key:
        return env_key

    if token_is_placeholder:
        return None
    return token


def _build_openai_client(
    api_key: str,
    timeout: int,
    base_url: Optional[str] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
):
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    return OpenAI(
        api_key=api_key,
        base_url=base_url or DEFAULT_OPENROUTER_BASE_URL,
        timeout=timeout,
        default_headers=headers or None,
    )


def api_models_map(model_name_or_path=None, token=None, **kwargs):
    if not model_name_or_path:
        return None
    token = _resolve_api_key(token)
    if not token:
        return None

    name = model_name_or_path.lower()

    if "gpt" in name:
        return GPT(model_name_or_path, token, **kwargs)
    if "claude" in name:
        return Claude45(model_name_or_path, token, **kwargs)
    if "gemini" in name:
        return Gemini(model_name_or_path, token, **kwargs)
    if "mistral" in name:
        return Mistral(model_name_or_path, token, **kwargs)
    if "deepseek" in name:
        return DeepSeek(model_name_or_path, token, **kwargs)
    if "qwen" in name:
        return Qwen(model_name_or_path, token, **kwargs)
    if "gemma" in name:
        return Gemma(model_name_or_path, token, **kwargs)
    if "grok" in name:
        return Grok(model_name_or_path, token, **kwargs)
    if "kimi" in name:
        return Kimi(model_name_or_path, token, **kwargs)
    if "llama" in name:
        return Llama(model_name_or_path, token, **kwargs)

    return None


class OpenAIChatModel:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 3  # initial try + up to 2 retries
    API_TIMEOUT = 60
    EMPTY_RETRYABLE_FINISH_REASONS = {"stop", "length"}
    EMPTY_NON_RETRYABLE_FINISH_REASONS = {"content_filter", "blocked"}
    ABNORMAL_OUTPUT_PATTERNS = (
        "<｜placeholder▁no▁",
        "<|placeholder",
        "placeholder▁no▁",
    )
    SCRIPT_MIX_COUNT_THRESHOLD = 10
    SCRIPT_MIX_MIN_FAMILIES = 3

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        extra_body: Optional[dict] = None,
        reasoning_enabled: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        api_query_sleep: Optional[float] = None,
        api_retry_sleep: Optional[float] = None,
        api_max_retry: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.base_url = base_url or DEFAULT_OPENROUTER_BASE_URL
        self.site_url = site_url
        self.site_name = site_name
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        # Runtime tunables (can also be overridden by env vars).
        self.api_query_sleep = float(
            api_query_sleep
            if api_query_sleep is not None
            else os.getenv("API_QUERY_SLEEP", self.API_QUERY_SLEEP)
        )
        self.api_retry_sleep = float(
            api_retry_sleep
            if api_retry_sleep is not None
            else os.getenv("API_RETRY_SLEEP", self.API_RETRY_SLEEP)
        )
        self.api_max_retry = int(
            api_max_retry
            if api_max_retry is not None
            else os.getenv("API_MAX_RETRY", self.API_MAX_RETRY)
        )
        self.max_concurrency = max(
            1,
            int(
                max_concurrency
                if max_concurrency is not None
                else os.getenv("API_MAX_CONCURRENCY", "1")
            ),
        )
        self.client = _build_openai_client(
            api_key=api_key,
            timeout=self.API_TIMEOUT,
            base_url=base_url,
            site_url=site_url,
            site_name=site_name,
        )
        self._last_generation_metadata = []

    def get_last_generation_metadata(self):
        return list(self._last_generation_metadata)

    @staticmethod
    def _extract_response_metadata(response):
        metadata = {
            "response_id": None,
            "response_model": None,
            "response_created": None,
            "provider": None,
            "system_fingerprint": None,
        }
        try:
            metadata["response_id"] = getattr(response, "id", None)
            metadata["response_model"] = getattr(response, "model", None)
            metadata["response_created"] = getattr(response, "created", None)
            metadata["system_fingerprint"] = getattr(response, "system_fingerprint", None)
        except Exception:
            pass

        try:
            raw = response.model_dump(mode="python")
        except Exception:
            raw = {}

        if metadata["response_id"] is None:
            metadata["response_id"] = raw.get("id")
        if metadata["response_model"] is None:
            metadata["response_model"] = raw.get("model")
        if metadata["response_created"] is None:
            metadata["response_created"] = raw.get("created")
        if metadata["system_fingerprint"] is None:
            metadata["system_fingerprint"] = raw.get("system_fingerprint")

        metadata["provider"] = raw.get("provider") or ((raw.get("usage") or {}).get("provider"))
        return metadata

    def _generate(
        self,
        prompt,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prompt_index: Optional[int] = None,
        **kwargs,
    ):
        output = self.API_ERROR_OUTPUT
        request_meta = {
            "prompt_index": prompt_index,
            "status": "error",
            "retries_used": 0,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "configured_model": self.model_name,
            "base_url": self.base_url,
            "site_url": self.site_url,
            "site_name": self.site_name,
            "reasoning_enabled": self.reasoning_enabled,
            "reasoning_effort": self.reasoning_effort,
            "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        for attempt in range(self.api_max_retry):
            retries_used = attempt
            request_meta["retries_used"] = retries_used
            try:
                request_messages = None
                prompt_level_extra_body = {}
                prompt_level_reasoning_enabled = None
                prompt_level_reasoning_effort = None

                if isinstance(prompt, list):
                    request_messages = prompt
                elif isinstance(prompt, dict):
                    if isinstance(prompt.get("messages"), list):
                        request_messages = prompt["messages"]
                    elif "role" in prompt and "content" in prompt:
                        request_messages = [prompt]
                    else:
                        request_messages = [{"role": "user", "content": str(prompt)}]

                    if isinstance(prompt.get("extra_body"), dict):
                        prompt_level_extra_body.update(prompt["extra_body"])
                    if "reasoning_enabled" in prompt:
                        prompt_level_reasoning_enabled = prompt.get("reasoning_enabled")
                    if "reasoning_effort" in prompt:
                        prompt_level_reasoning_effort = prompt.get("reasoning_effort")
                else:
                    request_messages = [{"role": "user", "content": prompt}]

                request_extra_body = {}
                request_extra_body.update(self.extra_body)
                request_extra_body.update(prompt_level_extra_body)

                override_extra_body = kwargs.get("extra_body")
                if isinstance(override_extra_body, dict):
                    request_extra_body.update(override_extra_body)

                effective_reasoning_enabled = kwargs.get("reasoning_enabled", self.reasoning_enabled)
                effective_reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
                if prompt_level_reasoning_enabled is not None:
                    effective_reasoning_enabled = prompt_level_reasoning_enabled
                if prompt_level_reasoning_effort:
                    effective_reasoning_effort = prompt_level_reasoning_effort

                # GPT-5.1: disabling thinking is most reliably expressed as reasoning.effort="none".
                model_lc = str(self.model_name).lower()
                is_gpt_51 = model_lc in {"gpt-5.1", "openai/gpt-5.1"}
                if is_gpt_51 and effective_reasoning_enabled is False and not effective_reasoning_effort:
                    effective_reasoning_effort = "none"

                if effective_reasoning_enabled is not None:
                    request_extra_body.setdefault("reasoning", {})
                    request_extra_body["reasoning"]["enabled"] = bool(effective_reasoning_enabled)
                if effective_reasoning_effort:
                    request_extra_body.setdefault("reasoning", {})
                    request_extra_body["reasoning"]["effort"] = effective_reasoning_effort

                request_meta["request_extra_body"] = request_extra_body or None
                requested_stream = bool(kwargs.get("stream", False))
                request_meta["request_stream"] = requested_stream

                request_payload = dict(
                    model=self.model_name,
                    messages=request_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=requested_stream,
                )
                if request_extra_body:
                    request_payload["extra_body"] = request_extra_body

                response = self.client.chat.completions.create(**request_payload)
                finish_reason, tokens_completion, native_tokens_reasoning, streamed = self._extract_generation_debug_fields(response)
                request_meta.update({
                    "finish_reason": finish_reason,
                    "tokens_completion": tokens_completion,
                    "native_tokens_reasoning": native_tokens_reasoning,
                    "streamed": streamed,
                })
                request_meta.update(self._extract_response_metadata(response))

                message = self._extract_first_message(response)
                if message is None:
                    raw_error = None
                    try:
                        raw = response.model_dump(mode="python")
                        raw_error = raw.get("error")
                    except Exception:
                        raw_error = None
                    request_meta["status"] = "invalid_response"
                    request_meta["error_type"] = "InvalidResponse"
                    request_meta["error_message"] = "response.choices is missing or empty"
                    if raw_error is not None:
                        request_meta["provider_error"] = raw_error
                    print(
                        f"[api_models] invalid response; retry={retries_used}/{self.api_max_retry - 1} "
                        f"choices missing/empty provider_error={raw_error}"
                    )
                    output = self.API_ERROR_OUTPUT
                    if attempt < self.api_max_retry - 1:
                        time.sleep(self.api_retry_sleep)
                        continue
                    break

                output, is_igr = self._coerce_message_content(getattr(message, "content", None))
                reasoning_details = getattr(message, "reasoning_details", None)
                request_meta["has_reasoning_details"] = reasoning_details is not None
                if is_igr:
                    content_type = type(getattr(message, "content", None)).__name__
                    request_meta["status"] = "igr_invalid_content_type"
                    request_meta["error_type"] = "IGR"
                    request_meta["error_message"] = f"Unsupported message.content type: {content_type}"
                    print(
                        f"[api_models] IGR invalid content type={content_type}; "
                        f"retry={retries_used}/{self.api_max_retry - 1}"
                    )
                    output = self.API_ERROR_OUTPUT
                    if attempt < self.api_max_retry - 1:
                        time.sleep(self.api_retry_sleep)
                        continue
                    break
                if isinstance(output, str):
                    output = output.strip()

                if not output:
                    should_retry_empty = self._should_retry_empty_output(
                        tokens_completion=tokens_completion,
                        finish_reason=finish_reason,
                        streamed=streamed,
                    )
                    print(
                        f"[api_models] empty content; retry={retries_used}/{self.api_max_retry - 1} "
                        f"finish_reason={finish_reason} tokens_completion={tokens_completion} "
                        f"native_tokens_reasoning={native_tokens_reasoning} streamed={streamed} "
                        f"should_retry={should_retry_empty}"
                    )
                    output = self.API_ERROR_OUTPUT
                    request_meta["status"] = "empty_output"
                    if should_retry_empty and attempt < self.api_max_retry - 1:
                        time.sleep(self.api_retry_sleep)
                        continue
                    break

                abnormal_reason = self._detect_abnormal_output_reason(output)
                if abnormal_reason:
                    request_meta["status"] = "igr_abnormal_output"
                    request_meta["error_type"] = "IGR"
                    request_meta["error_message"] = abnormal_reason
                    print(
                        f"[api_models] abnormal output detected ({abnormal_reason}); "
                        f"retry={retries_used}/{self.api_max_retry - 1}"
                    )
                    output = self.API_ERROR_OUTPUT
                    if attempt < self.api_max_retry - 1:
                        time.sleep(self.api_retry_sleep)
                        continue
                    break

                print(
                    f"[api_models] success; retry={retries_used}/{self.api_max_retry - 1} "
                    f"finish_reason={finish_reason} tokens_completion={tokens_completion} "
                    f"native_tokens_reasoning={native_tokens_reasoning} streamed={streamed}"
                )
                request_meta["status"] = "success"
                if kwargs.get("return_assistant_message"):
                    assistant_message = {"role": "assistant", "content": output}
                    if reasoning_details is not None:
                        assistant_message["reasoning_details"] = reasoning_details
                    request_meta["assistant_message_returned"] = True
                    output = assistant_message
                break
            except openai.OpenAIError as e:
                request_meta["status"] = "api_error"
                request_meta["error_type"] = type(e).__name__
                request_meta["error_message"] = str(e)
                print(f"[api_models] OpenAIError; retry={retries_used}/{self.api_max_retry - 1} {type(e)} {e}")
                if attempt < self.api_max_retry - 1:
                    retry_sleep = float(self.api_retry_sleep)
                    # Scope smarter backoff only to DeepSeek v3.2 to avoid changing global behavior.
                    if self._is_deepseek_v32():
                        retry_sleep = self._compute_deepseek_v32_retry_sleep(attempt=attempt, exc=e)
                    request_meta["retry_sleep_sec"] = retry_sleep
                    time.sleep(retry_sleep)
            except Exception as e:
                request_meta["status"] = "unexpected_error"
                request_meta["error_type"] = type(e).__name__
                request_meta["error_message"] = str(e)
                print(f"[api_models] UnexpectedError; retry={retries_used}/{self.api_max_retry - 1} {type(e)} {e}")
                if attempt < self.api_max_retry - 1:
                    time.sleep(self.api_retry_sleep)
            time.sleep(self.api_query_sleep)

        request_meta["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._last_generation_metadata.append(request_meta)
        return output

    def _is_deepseek_v32(self) -> bool:
        return str(self.model_name).strip().lower() == "deepseek/deepseek-v3.2"

    def _compute_deepseek_v32_retry_sleep(self, attempt: int, exc: Exception) -> float:
        max_sleep = 120.0
        retry_after = self._extract_retry_after_seconds(exc)
        if retry_after is not None:
            # Respect explicit provider/API throttle windows when available.
            return max(self.api_query_sleep, min(float(retry_after), max_sleep))

        is_rate_limit = isinstance(exc, openai.RateLimitError) or "429" in str(exc)
        if is_rate_limit:
            # DeepSeek v3.2-specific exponential backoff + jitter for transient 429 bursts.
            base = max(2.0, float(self.api_retry_sleep))
            jitter = random.uniform(0.0, 1.5)
            backoff = base * (2 ** attempt) + jitter
            return min(backoff, max_sleep)

        return max(float(self.api_retry_sleep), float(self.api_query_sleep))

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        retry_after = None
        if isinstance(headers, dict):
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        else:
            retry_after = (
                getattr(headers, "get", lambda *_: None)("retry-after")
                or getattr(headers, "get", lambda *_: None)("Retry-After")
            )
        if retry_after is None:
            return None

        try:
            value = float(retry_after)
            return value if value >= 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_first_message(response):
        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                return getattr(choices[0], "message", None)
        except Exception:
            pass
        return None

    @staticmethod
    def _coerce_message_content(content):
        if content is None:
            return None, False
        if isinstance(content, str):
            return content, False
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") != "text":
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                else:
                    if getattr(item, "type", None) != "text":
                        continue
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
            if parts:
                return "".join(parts), False
            return None, False
        # Any non-standard content payload is treated as Invalid Generation Response (IGR).
        return None, True

    @staticmethod
    def _extract_generation_debug_fields(response):
        finish_reason = None
        tokens_completion = None
        native_tokens_reasoning = None
        streamed = None

        try:
            if response.choices:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
        except Exception:
            pass

        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_completion = getattr(usage, "completion_tokens", None)
                completion_details = getattr(usage, "completion_tokens_details", None)
                if completion_details is not None:
                    native_tokens_reasoning = (
                        getattr(completion_details, "reasoning_tokens", None)
                        or getattr(completion_details, "native_tokens_reasoning", None)
                    )
        except Exception:
            pass

        try:
            raw = response.model_dump(mode="python")
        except Exception:
            raw = {}

        if finish_reason is None:
            finish_reason = (
                raw.get("finish_reason")
                or ((raw.get("choices") or [{}])[0].get("finish_reason"))
            )
        if tokens_completion is None:
            tokens_completion = raw.get("tokens_completion") or (raw.get("usage") or {}).get("completion_tokens")
        if native_tokens_reasoning is None:
            native_tokens_reasoning = (
                raw.get("native_tokens_reasoning")
                or (raw.get("usage") or {}).get("native_tokens_reasoning")
                or ((raw.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
            )
        streamed = getattr(response, "streamed", None) if streamed is None else streamed
        if streamed is None:
            streamed = raw.get("streamed")

        return finish_reason, tokens_completion, native_tokens_reasoning, streamed

    @classmethod
    def _should_retry_empty_output(cls, tokens_completion, finish_reason, streamed):
        finish_reason_norm = str(finish_reason).strip().lower() if finish_reason is not None else None
        tokens_completion_int = None
        try:
            tokens_completion_int = int(tokens_completion) if tokens_completion is not None else None
        except (TypeError, ValueError):
            tokens_completion_int = None

        # Explicit provider-side block/filter: do not retry blindly.
        if (
            tokens_completion_int == 0
            and finish_reason_norm in cls.EMPTY_NON_RETRYABLE_FINISH_REASONS
        ):
            return False

        # Retry only when an empty output appears likely to be a transport/aggregation issue.
        return (
            (tokens_completion_int is not None and tokens_completion_int > 0)
            or (finish_reason_norm in cls.EMPTY_RETRYABLE_FINISH_REASONS)
            or (streamed is True)
        )

    @classmethod
    def _detect_abnormal_output_reason(cls, output):
        if not isinstance(output, str):
            return "non_string_output_after_parse"

        lowered = output.lower()
        for pattern in cls.ABNORMAL_OUTPUT_PATTERNS:
            if pattern.lower() in lowered:
                return f"placeholder_token_detected:{pattern}"

        # Replacement char usually indicates decoding/transport corruption.
        if "\ufffd" in output:
            return "replacement_character_detected"

        script_counts = cls._count_script_families(output)
        active_families = [name for name, count in script_counts.items() if count > cls.SCRIPT_MIX_COUNT_THRESHOLD]
        if len(active_families) >= cls.SCRIPT_MIX_MIN_FAMILIES:
            details = ",".join(f"{k}:{script_counts[k]}" for k in active_families)
            return f"mixed_script_noise_detected:{details}"

        return None

    @staticmethod
    def _count_script_families(text):
        counts = {
            "latin": 0,
            "cyrillic": 0,
            "arabic": 0,
            "cjk": 0,
        }
        for ch in text:
            code = ord(ch)
            if (
                0x0041 <= code <= 0x005A  # A-Z
                or 0x0061 <= code <= 0x007A  # a-z
                or 0x00C0 <= code <= 0x024F  # Latin-1 Supplement + Extended-A/B
                or 0x1E00 <= code <= 0x1EFF  # Latin Extended Additional
            ):
                counts["latin"] += 1
            elif 0x0400 <= code <= 0x052F:  # Cyrillic + supplement
                counts["cyrillic"] += 1
            elif (
                0x0600 <= code <= 0x06FF  # Arabic
                or 0x0750 <= code <= 0x077F  # Arabic Supplement
                or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
                or 0xFB50 <= code <= 0xFDFF  # Arabic Presentation Forms-A
                or 0xFE70 <= code <= 0xFEFF  # Arabic Presentation Forms-B
            ):
                counts["arabic"] += 1
            elif (
                0x3400 <= code <= 0x4DBF  # CJK Extension A
                or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
                or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
                or 0x3040 <= code <= 0x309F  # Hiragana
                or 0x30A0 <= code <= 0x30FF  # Katakana
                or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
            ):
                counts["cjk"] += 1
        return counts

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        use_tqdm: bool = False,
        **kwargs,
    ):
        self._last_generation_metadata = []
        if self.max_concurrency <= 1:
            prompt_iter = tqdm(prompts) if use_tqdm else prompts
            return [
                self._generate(prompt, max_new_tokens, temperature, top_p, prompt_index=i, **kwargs)
                for i, prompt in enumerate(prompt_iter)
            ]

        outputs = [self.API_ERROR_OUTPUT] * len(prompts)
        iterator = range(len(prompts))
        progress = tqdm(total=len(prompts)) if use_tqdm else None
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futures = {
                ex.submit(
                    self._generate,
                    prompts[i],
                    max_new_tokens,
                    temperature,
                    top_p,
                    i,
                    **kwargs,
                ): i
                for i in iterator
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    outputs[idx] = fut.result()
                except Exception:
                    outputs[idx] = self.API_ERROR_OUTPUT
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
        return outputs


class GPT(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        normalized_model_name = model_name.strip() if isinstance(model_name, str) else model_name
        normalized_model_name_lc = str(normalized_model_name).lower()

        # Keep gpt-oss-120b pinned to DeepInfra on OpenRouter (no fallback).
        if normalized_model_name_lc in {
            "gpt-oss-120b",
            "openai/gpt-oss-120b",
        }:
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            forced_provider_policy = {
                "provider": {
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        # Keep selected OpenAI GPT models pinned to provider OpenAI (no fallback).
        elif normalized_model_name_lc in {
            "gpt-5.1",
            "openai/gpt-5.1",
            "gpt-5.2-chat",
            "openai/gpt-5.2-chat",
            "gpt-4o-2024-11-20",
            "openai/gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "openai/gpt-4o-mini",
        }:
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            forced_provider_policy = {
                "provider": {
                    "order": ["OpenAI"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        super().__init__(
            model_name=normalized_model_name,
            api_key=api_key,
            **kwargs,
        )


class Claude45(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        normalized_model_name = _normalize_openrouter_claude_model_name(model_name)
        normalized_model_name_lc = str(normalized_model_name).lower()

        # Keep all Claude models pinned to Anthropic on OpenRouter (no fallback).
        if normalized_model_name_lc.startswith("anthropic/claude-"):
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            forced_provider_policy = {
                "provider": {
                    "order": ["Anthropic"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        super().__init__(
            model_name=normalized_model_name,
            api_key=api_key,
            **kwargs,
        )


class Gemini(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        normalized_model_name = _normalize_openrouter_gemini_model_name(model_name)
        normalized_model_name_lc = str(normalized_model_name).lower()

        # Keep all Gemini models pinned to Google AI Studio on OpenRouter (no fallback).
        if normalized_model_name_lc.startswith("google/gemini-"):
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            forced_provider_policy = {
                "provider": {
                    "order": ["Google AI Studio"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        super().__init__(
            model_name=normalized_model_name,
            api_key=api_key,
            **kwargs,
        )


class Mistral(OpenAIChatModel):
    pass


class DeepSeek(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        normalized_model_name = _normalize_openrouter_deepseek_model_name(model_name)
        # Keep deepseek-v3.2 pinned to DeepInfra on OpenRouter (no provider fallback).
        if normalized_model_name.lower() == "deepseek/deepseek-v3.2":
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            # Scope retry-strengthening only to DeepSeek v3.2.
            kwargs.setdefault("api_max_retry", int(os.getenv("DEEPSEEK_V32_API_MAX_RETRY", "6")))
            kwargs.setdefault("api_retry_sleep", float(os.getenv("DEEPSEEK_V32_API_RETRY_SLEEP", "10")))
            forced_provider_policy = {
                "provider": {
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        super().__init__(
            model_name=normalized_model_name,
            api_key=api_key,
            **kwargs,
        )


class Qwen(OpenAIChatModel):
    # DashScope-specific logic removed: API experiments now use OpenRouter uniformly.
    pass


class Gemma(OpenAIChatModel):
    pass


class Grok(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(
            model_name=_normalize_openrouter_grok_model_name(model_name),
            api_key=api_key,
            **kwargs,
        )


class Kimi(OpenAIChatModel):
    pass


class Llama(OpenAIChatModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        normalized_model_name = model_name.strip() if isinstance(model_name, str) else model_name
        normalized_model_name_lc = str(normalized_model_name).lower()

        # Keep llama-3.3-70b-instruct pinned to Inceptron on OpenRouter (no fallback).
        if normalized_model_name_lc in {
            "meta-llama/llama-3.3-70b-instruct",
            "llama-3.3-70b-instruct",
        }:
            user_extra_body = kwargs.get("extra_body", {})
            if not isinstance(user_extra_body, dict):
                user_extra_body = {}
            forced_provider_policy = {
                "provider": {
                    "order": ["Inceptron"],
                    "allow_fallbacks": False,
                }
            }
            kwargs["extra_body"] = _deep_merge_dict(user_extra_body, forced_provider_policy)

        super().__init__(
            model_name=normalized_model_name,
            api_key=api_key,
            **kwargs,
        )
