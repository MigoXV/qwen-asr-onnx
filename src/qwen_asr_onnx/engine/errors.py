class EngineError(RuntimeError):
    """内部引擎错误基类。"""


class EngineUnavailableError(EngineError):
    """引擎未就绪、已关闭或 Worker 故障。"""


class EngineResourceExhaustedError(EngineError):
    """引擎达到显式容量上限。"""


class EngineDeadlineExceededError(EngineError):
    """请求在排队或执行期间超过 deadline。"""
