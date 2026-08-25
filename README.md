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
- `AxWorker` 在一个专用线程中串行执行模型加载、预热、推理和关闭，并把 native token 事件安全投递回 asyncio 事件循环。
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
  --timeout 120 \
  --interim-results
```

`interim_results=true` 时，decoder 每生成一个有效文本 token，服务就返回一条累计 transcript，`is_final=false`；生成结束后再返回完整 transcript，`is_final=true`。设置 `--no-interim-results` 时保持单条 final 响应。任何响应都不会暴露 `language Chinese<asr_text>` 等模型内部前缀。hotwords 字段仍兼容接收，但当前固定 AX prompt 不消费它。

需要观察响应到达时间时增加 `--show-timing`。时间和 final 标志写入 stderr，stdout 仍只包含干净 transcript。

## 健康检查

服务注册了标准 gRPC Health 协议：

- 方法：`grpc.health.v1.Health/Check`
- 服务名：`ux_speech.UxSpeech`
- 模型加载和预热期间：`NOT_SERVING`
- Engine 就绪后：`SERVING`
- Engine 失败或服务退出：`NOT_SERVING`

因此部署探针无需修改 ASR 通讯协议，也无需额外定义私有健康接口。

## 并发和取消语义

AX runner 当前为单模型、单 worker 串行执行。asyncio 服务可以并发接收客户端事件和持续发送 token 文本，但实际 native 推理由专用工作线程串行调度：

- 超过 `QWEN_ASR_MAX_INFLIGHT_REQUESTS` 的请求立即返回 `RESOURCE_EXHAUSTED`；
- 排队请求在客户端取消或 deadline 到期后不会进入 native 推理；
- 已进入 native 调用的请求会在下一个 token callback 边界观察取消或超时并停止生成；正在执行的单次 NPU decode 无法从中间打断；
- 服务关闭时先停止接收新请求，再按宽限期排空，最后在 worker 线程关闭 runtime。

## 开发验证

```bash
poetry check
poetry run python -m compileall -q src examples tests
poetry run pytest -q
git diff --check
```

AX native runner 的构建、固定模型约束和板端基准说明见 [`src/qwen_asr_onnx/ax_m4c/README.md`](src/qwen_asr_onnx/ax_m4c/README.md)。

## AX650 部署镜像

使用官方 AX650 模型时，构建独立的 ARM64 部署镜像：

```bash
docker build \
  -f docker/Dockerfile.ax \
  -t registry.cn-hangzhou.aliyuncs.com/migo-dl/qwen-asr-ax:0.1.0a1-aarch64 \
  .
```

模型不写入镜像，启动时将官方模型目录只读挂载到 `/models`：

```bash
docker run --rm --network host --privileged \
  -e MODEL_PATH=/models \
  -v /data/repositories/model-bin/AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047:/models:ro \
  registry.cn-hangzhou.aliyuncs.com/migo-dl/qwen-asr-ax:0.1.0a1-aarch64
```

容器默认执行 `qwen-asr serve`，监听 `50051`，并在模型加载与预热完成后通过标准 gRPC Health 接口报告 `SERVING`。
