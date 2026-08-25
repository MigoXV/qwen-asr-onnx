# qwen-asr-onnx

Qwen3-ASR-0.6B 的 AX650/MC50 gRPC 推理服务。项目名为历史名称，当前主链路不再包含 ONNX Runtime：

```text
UxSpeech gRPC -> ASRServicer -> AxInferenceEngine
              -> AxQwenAsr/CFFI -> AXEngine
```

AX runner 固定支持 C64/P448/CTX2047、BF16 activation、16 kHz 单声道 PCM16。单个进程只持有一个 AX handle，由专用线程串行执行；默认最多接受两个 inflight 请求，超过容量返回 gRPC `RESOURCE_EXHAUSTED`。

## 启动服务

先按 [AX650 专用后端说明](src/qwen_asr_onnx/ax_m4c/README.md) 构建并安装 `libax_qwen_asr.so`，然后运行：

```bash
export LD_LIBRARY_PATH=/soc/lib
export MODEL_PATH=/workspace/model-bin/AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047

poetry run python -m qwen_asr_onnx.commands.app serve \
  --config examples/ax-m4c/grpc.yaml
```

也可以使用 Poetry 脚本入口：

```bash
poetry run qwen-asr serve --config examples/ax-m4c/grpc.yaml
```

服务在模型加载和预热成功后才开始监听业务请求。配置字段见 [grpc.yaml](examples/ax-m4c/grpc.yaml)。

## 调用服务

协议保持为 `ux_speech.UxSpeech/StreamingRecognize`：第一包发送 `streaming_config`，第二包发送一段完整 PCM16 音频。当前是离线整段识别语义，只返回一个 `is_final=true` 的响应，不产生中间事件。

```bash
poetry run python examples/ax-m4c/grpc_client.py \
  data-bin/2026-08-24/audio/6259c2f6915b_011.wav \
  --target 127.0.0.1:50051
```

模型原始输出中的 `language Chinese<asr_text>` 等协议前缀会在服务端清理；客户端收到的 `alternative.transcript` 只有正文，语言写入 `alternative.language.code`。

兼容说明：AX 固定 prompt 暂不支持 `context` 和 `hotwords`，服务会忽略并记录告警；请求 `interim_results=true` 时仍只返回最终结果。每次请求最多包含 959,998 字节 PCM，约 30 秒音频。

## 验证

```bash
poetry run pytest -q -m "not ax650"

AX_QWEN_ASR_MODEL_DIR="$MODEL_PATH" \
AX_QWEN_ASR_AUDIO=data-bin/rita.wav \
poetry run pytest -q -s -m ax650 tests/ax_m4c/test_integration_ax650.py
```
