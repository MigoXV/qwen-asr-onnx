# Qwen3-ASR AX650 gRPC 推理服务

本项目把 Qwen3-ASR-0.6B 的 AX650 native 推理链路封装为异步 gRPC 服务。主服务不再使用 ONNX，也不再依赖 OmegaConf 或 YAML 配置；服务参数统一由 Typer 命令行和环境变量提供。

现有 AX native 算法保持不变，服务侧只负责模型生命周期、请求排队、异步调用、限流、超时、健康检查和结构化日志。

## 架构

```text
gRPC async client stream
    -> grpc.aio ASRServicer
        -> AsrInferencer
            -> EngineClient / EngineCore
                -> bounded request queue
                    -> AxWorker（专用工作线程）
                        -> AxRuntime
                            -> AxModelRunner
                                -> AxQwenAsr / CFFI
                                    -> libax_qwen_asr.so / AXEngine
```

- `ASRServicer` 保留现有 `ux_speech.UxSpeech` 通讯协议，处理请求校验和 gRPC 状态码。
- `EngineCore` 是外层异步引擎，管理请求 ID、容量、deadline、取消和优雅退出。
- `AxWorker` 在一个专用线程中串行执行模型加载、预热、推理和关闭，避免阻塞 asyncio 事件循环。
- `AxRuntime` 管理后端状态，`AxModelRunner` 适配公开的 `AxQwenAsr`。
- `AxQwenAsr` 及其 native 实现仍是原来的 AX 推理算法入口。

## 安装

项目使用 Poetry：

```bash
poetry install
```

AXEngine 动态库不在系统默认搜索路径时，需要在启动进程前配置：

```bash
export LD_LIBRARY_PATH=/soc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## 启动服务

全部服务配置都可以通过环境变量提供：

```bash
export LD_LIBRARY_PATH=/soc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MODEL_PATH=/workspace/model-bin/AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047
export GRPC_HOST='[::]'
export GRPC_PORT=50051
export QWEN_ASR_AX_WARMUP=true
export QWEN_ASR_MAX_INFLIGHT_REQUESTS=2
export QWEN_ASR_SHUTDOWN_GRACE_SECONDS=5
export LOG_LEVEL=INFO

poetry run qwen-asr serve
```

配置项如下：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | 无，必填 | AX650 模型根目录 |
| `GRPC_HOST` | `[::]` | gRPC 监听地址 |
| `GRPC_PORT` | `50051` | gRPC 监听端口 |
| `QWEN_ASR_AX_WARMUP` | `true` | 对外就绪前是否预热模型 |
| `QWEN_ASR_MAX_INFLIGHT_REQUESTS` | `2` | 执行中与排队请求的总容量 |
| `QWEN_ASR_SHUTDOWN_GRACE_SECONDS` | `5.0` | 服务关闭时的请求排空宽限期 |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

同一组选项也能显式传给命令行，可通过以下命令查看：

```bash
poetry run qwen-asr serve --help
```

仓库提供了 [`.env.example`](.env.example)。Typer 会读取已经存在于进程环境中的变量，但不会在普通 shell 中自动加载 `.env` 文件；需要时可以执行：

```bash
set -a
source .env
set +a
poetry run qwen-asr serve
```

## gRPC 调用

通讯协议保持为 `ux_speech.UxSpeech/StreamingRecognize`。客户端先发送一个 `streaming_config`，再发送一个完整的 16 kHz、单声道 PCM16 音频包：

```bash
poetry run python examples/ax-m4c/grpc_client.py \
  data-bin/2026-08-24/audio/6259c2f6915b_011.wav \
  --target 127.0.0.1:50051 \
  --timeout 120
```

服务只返回清理后的最终转写文本，不会把 `language Chinese<asr_text>` 等模型内部前缀暴露给客户端。现有协议字段中的 hotwords 和 interim 配置仍可发送，但当前 AX 算法不消费它们。

## 健康检查

服务注册了标准 gRPC Health 协议：

- 方法：`grpc.health.v1.Health/Check`
- 服务名：`ux_speech.UxSpeech`
- 模型加载和预热期间：`NOT_SERVING`
- Engine 就绪后：`SERVING`
- Engine 失败或服务退出：`NOT_SERVING`

因此部署探针无需修改 ASR 通讯协议，也无需额外定义私有健康接口。

## 并发和取消语义

AX runner 当前为单模型、单 worker 串行执行。asyncio 服务可以并发接收客户端事件，但实际 native 推理由专用工作线程串行调度：

- 超过 `QWEN_ASR_MAX_INFLIGHT_REQUESTS` 的请求立即返回 `RESOURCE_EXHAUSTED`；
- 排队请求在客户端取消或 deadline 到期后不会进入 native 推理；
- 已进入 native 调用的请求无法强制中断，结果会在取消或超时后丢弃；
- 服务关闭时先停止接收新请求，再按宽限期排空，最后在 worker 线程关闭 runtime。

## 开发验证

```bash
poetry check
poetry run python -m compileall -q src examples tests
poetry run pytest -q
git diff --check
```

AX native runner 的构建、固定模型约束和板端基准说明见 [`src/qwen_asr_onnx/ax_m4c/README.md`](src/qwen_asr_onnx/ax_m4c/README.md)。
