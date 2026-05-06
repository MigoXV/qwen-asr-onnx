docker run -it --rm \
-p 50051:50051 \
-e QWEN_ASR_CONFIG=examples/qwen-asr-a733/config.yaml \
-v ${PWD}/model-bin:/app/model-bin \
registry.cn-hangzhou.aliyuncs.com/migo-dl/qwen-asr-onnx:0.1.0a1-amd64