# coding=utf-8
from __future__ import annotations

from typing import Optional, Tuple

from qwen_asr_onnx.inferencers.language import normalize_language_name

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    def fix_char_repeats(value: str, thresh: int) -> str:
        result: list[str] = []
        index = 0
        length = len(value)
        while index < length:
            count = 1
            while index + count < length and value[index + count] == value[index]:
                count += 1

            if count > thresh:
                result.append(value[index])
                index += count
            else:
                result.append(value[index : index + count])
                index += count
        return "".join(result)

    def fix_pattern_repeats(value: str, thresh: int, max_len: int = 20) -> str:
        length = len(value)
        min_repeat_chars = thresh * 2
        if length < min_repeat_chars:
            return value

        index = 0
        result: list[str] = []
        found = False
        while index <= length - min_repeat_chars:
            found = False
            for pattern_len in range(1, max_len + 1):
                if index + pattern_len * thresh > length:
                    break

                pattern = value[index : index + pattern_len]
                valid = True
                for repeat in range(1, thresh):
                    start = index + repeat * pattern_len
                    if value[start : start + pattern_len] != pattern:
                        valid = False
                        break

                if valid:
                    end_index = index + thresh * pattern_len
                    while (
                        end_index + pattern_len <= length
                        and value[end_index : end_index + pattern_len] == pattern
                    ):
                        end_index += pattern_len
                    result.append(pattern)
                    result.append(fix_pattern_repeats(value[end_index:], thresh, max_len))
                    index = length
                    found = True
                    break

            if found:
                break
            result.append(value[index])
            index += 1

        if not found:
            result.append(value[index:])
        return "".join(result)

    fixed_text = fix_char_repeats(text, threshold)
    return fix_pattern_repeats(fixed_text, threshold)


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    将 Qwen3-ASR 原始输出解析为 ``(language, text)``。

    支持 ``language Chinese<asr_text>...``、带换行元信息、无标签纯文本，以及
    ``language None<asr_text>`` 空音频输出。提供 ``user_language`` 时会强制返回该
    语言，同时防御性剥离模型回显的协议前缀。
    """
    if raw is None:
        return "", ""
    value = str(raw).strip()
    if not value:
        return "", ""

    value = detect_and_fix_repetitions(value)

    if user_language:
        text = value
        if _ASR_TEXT_TAG in text:
            _, text = text.split(_ASR_TEXT_TAG, 1)
        else:
            forced_prefix = f"{_LANG_PREFIX}{user_language}"
            if text.lower().startswith(forced_prefix.lower()):
                text = text[len(forced_prefix) :]
        return user_language, text.strip()

    meta_part = value
    text_part = ""
    if _ASR_TEXT_TAG in value:
        meta_part, text_part = value.split(_ASR_TEXT_TAG, 1)
    else:
        return "", value.strip()

    meta_lower = meta_part.lower()
    if "language none" in meta_lower:
        text = text_part.strip()
        if not text:
            return "", ""
        return "", text

    language = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(_LANG_PREFIX):
            detected = line[len(_LANG_PREFIX) :].strip()
            if detected:
                language = normalize_language_name(detected)
            break

    return language, text_part.strip()
