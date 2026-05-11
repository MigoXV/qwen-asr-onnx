# coding=utf-8
from __future__ import annotations

SUPPORTED_LANGUAGES: list[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]

LANGUAGE_CODE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-sg": "Chinese",
    "zh-tw": "Chinese",
    "zh-hant": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "en-au": "English",
    "en-in": "English",
    "english": "English",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "zh-yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "ar-sa": "Arabic",
    "ar-eg": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "de-de": "German",
    "de-at": "German",
    "de-ch": "German",
    "german": "German",
    "fr": "French",
    "fr-fr": "French",
    "fr-ca": "French",
    "french": "French",
    "es": "Spanish",
    "es-es": "Spanish",
    "es-mx": "Spanish",
    "es-ar": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "id-id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "ru-ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "th-th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vi-vn": "Vietnamese",
    "vietnamese": "Vietnamese",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    "tr": "Turkish",
    "tr-tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hi-in": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "ms-my": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "nl-nl": "Dutch",
    "nl-be": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "sv-se": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "da-dk": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "fi-fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "pl-pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "cs-cz": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "filipino": "Filipino",
    "tagalog": "Filipino",
    "fa": "Persian",
    "fa-ir": "Persian",
    "persian": "Persian",
    "farsi": "Persian",
    "el": "Greek",
    "el-gr": "Greek",
    "greek": "Greek",
    "ro": "Romanian",
    "ro-ro": "Romanian",
    "romanian": "Romanian",
    "hu": "Hungarian",
    "hu-hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "mk-mk": "Macedonian",
    "macedonian": "Macedonian",
}


def normalize_language_name(language: str) -> str:
    """将语言名称规范化为 Qwen3-ASR 使用的英文首字母大写形式。"""
    if language is None:
        raise ValueError("language is None")
    value = str(language).strip()
    if not value:
        raise ValueError("language is empty")
    return value[:1].upper() + value[1:].lower()


def resolve_language_code(code: str | None) -> str | None:
    """将 ISO/BCP-47 语言代码或英文语言名解析为 Qwen3-ASR 规范语言名。"""
    if code is None:
        return None
    key = str(code).strip()
    if not key:
        return None

    result = LANGUAGE_CODE_MAP.get(key.lower())
    if result is not None:
        return result

    normalized = normalize_language_name(key)
    if normalized in SUPPORTED_LANGUAGES:
        return normalized

    return None
