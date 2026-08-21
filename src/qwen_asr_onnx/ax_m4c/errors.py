"""AX Qwen3-ASR native 错误码到 Python 异常的映射。"""

from __future__ import annotations

from enum import IntEnum


class NativeErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    MODEL_CONFIG = 2
    MODEL_FILE = 3
    AXENGINE = 4
    OUT_OF_MEMORY = 5
    BUFFER_TOO_SMALL = 6
    BUSY = 7
    INTERNAL = 8


class AxQwenAsrError(RuntimeError):
    """所有 AX Qwen3-ASR 错误的基类。"""

    def __init__(self, message: str, *, code: int | NativeErrorCode | None = None):
        super().__init__(message)
        self.code = code


class AxQwenAsrLibraryError(AxQwenAsrError):
    """native 动态库不存在或无法加载。"""


class AxQwenAsrInvalidArgumentError(AxQwenAsrError, ValueError):
    """Python 参数或 C ABI 参数无效。"""


class AxQwenAsrModelError(AxQwenAsrError):
    """模型配置或模型文件不符合固定 profile。"""


class AxQwenAsrRuntimeError(AxQwenAsrError):
    """AXEngine、内存或 native 内部运行错误。"""


class AxQwenAsrBufferTooSmallError(AxQwenAsrError, BufferError):
    """调用者提供的输出缓冲区不足。"""

    def __init__(self, message: str, *, required_size: int, code: int):
        super().__init__(message, code=code)
        self.required_size = required_size


class AxQwenAsrBusyError(AxQwenAsrError):
    """进程内已有一个 runner，或同一 handle 正在执行请求。"""


ERROR_CLASS_BY_CODE: dict[NativeErrorCode, type[AxQwenAsrError]] = {
    NativeErrorCode.INVALID_ARGUMENT: AxQwenAsrInvalidArgumentError,
    NativeErrorCode.MODEL_CONFIG: AxQwenAsrModelError,
    NativeErrorCode.MODEL_FILE: AxQwenAsrModelError,
    NativeErrorCode.AXENGINE: AxQwenAsrRuntimeError,
    NativeErrorCode.OUT_OF_MEMORY: AxQwenAsrRuntimeError,
    NativeErrorCode.BUFFER_TOO_SMALL: AxQwenAsrBufferTooSmallError,
    NativeErrorCode.BUSY: AxQwenAsrBusyError,
    NativeErrorCode.INTERNAL: AxQwenAsrRuntimeError,
}
