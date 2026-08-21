#!/usr/bin/env python3
"""Run one Qwen3-ASR utterance with the AX650 AXEngine model."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import axengine
import librosa
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer


SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
MEL_FRAMES = 3_000
PREFILL_TOKENS = 64
MAX_CONTEXT = 2_047
MAX_PREFILL = 448

AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
NEWLINE_ID = 198


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("只支持 16-bit 单声道 PCM WAV")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"只支持 {SAMPLE_RATE} Hz WAV")
        pcm = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def make_mel(wav: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep this identical to the project's ONNX preprocessing."""
    mel_filters = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE // 2,
        norm="slaney",
        htk=False,
    ).astype(np.float32)
    stft = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    mel = mel_filters @ (np.abs(stft) ** 2)
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = ((mel + 4.0) / 4.0).astype(np.float32)
    valid_frames = mel.shape[1]
    if valid_frames > MEL_FRAMES:
        raise ValueError(f"音频过长：mel 帧数 {valid_frames} > {MEL_FRAMES}")
    padded = np.zeros((1, MEL_FRAMES, N_MELS), dtype=np.float32)
    padded[0, :valid_frames] = mel.T
    return padded, valid_frames


def downsampled_length(length: int) -> int:
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def encode_audio(model_dir: Path, wav: np.ndarray) -> np.ndarray:
    mel, mel_frames = make_mel(wav)
    conv = axengine.InferenceSession(model_dir / "conv_frontend.axmodel")
    conv_output = conv.run(None, {"input_features": mel})[0]
    audio_tokens = downsampled_length(mel_frames)
    mask = np.zeros((1, conv_output.shape[1]), dtype=np.uint8)
    mask[:, :audio_tokens] = 1
    encoder = axengine.InferenceSession(model_dir / "encoder.axmodel")
    features = encoder.run(
        None,
        {
            "input_features": conv_output,
            "feature_attention_mask": mask,
        },
    )[0]
    return features[0, :audio_tokens].astype(ml_dtypes.bfloat16)


def prompt_ids(tokenizer, audio_tokens: int) -> list[int]:
    ids = [IM_START_ID] + tokenizer.encode("system", add_special_tokens=False) + [NEWLINE_ID]
    ids += [IM_END_ID, NEWLINE_ID]
    ids += [IM_START_ID] + tokenizer.encode("user", add_special_tokens=False) + [NEWLINE_ID]
    ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * audio_tokens + [AUDIO_END_ID]
    ids += [IM_END_ID, NEWLINE_ID]
    ids += [IM_START_ID] + tokenizer.encode("assistant", add_special_tokens=False) + [NEWLINE_ID]
    return ids


def causal_mask(history: int, valid: int) -> np.ndarray:
    width = history + PREFILL_TOKENS
    mask = np.full((1, PREFILL_TOKENS, width), -65536, dtype=ml_dtypes.bfloat16)
    for row in range(valid):
        mask[0, row, :history] = 0
        mask[0, row, history : history + row + 1] = 0
    return mask


def transcribe(model_dir: Path, audio_path: Path, max_new_tokens: int) -> str:
    config = json.loads((model_dir / "config.json").read_text())
    layer_paths = [
        model_dir / (config["template_filename_axmodel"] % i)
        for i in range(config["axmodel_num"])
    ]
    required = [
        model_dir / "conv_frontend.axmodel",
        model_dir / "encoder.axmodel",
        model_dir / config["filename_post_axmodel"],
        model_dir / config["url_tokenizer_model"],
        *layer_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少模型文件：\n" + "\n".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    audio_features = encode_audio(model_dir, read_wav(audio_path))
    ids = prompt_ids(tokenizer, audio_features.shape[0])

    vocab_size = int(config["tokens_embed_num"])
    hidden_size = int(config["tokens_embed_size"])
    embedding_path = model_dir / config["filename_tokens_embed"]
    embeddings = np.memmap(
        embedding_path,
        dtype=ml_dtypes.bfloat16,
        mode="r",
        shape=(vocab_size, hidden_size),
    )
    hidden = np.array(embeddings[np.asarray(ids)], dtype=ml_dtypes.bfloat16, copy=True)
    audio_positions = np.flatnonzero(np.asarray(ids) == AUDIO_PAD_ID)
    if len(audio_positions) != len(audio_features):
        raise RuntimeError("音频特征数量与 AUDIO_PAD token 数量不一致")
    hidden[audio_positions] = audio_features

    if len(ids) > MAX_PREFILL:
        raise ValueError(f"prompt token 数 {len(ids)} 超过 P448 模型上限")

    print(f"音频特征 token: {len(audio_features)}，prompt token: {len(ids)}", flush=True)
    layers = [axengine.InferenceSession(path) for path in layer_paths]
    post = axengine.InferenceSession(model_dir / config["filename_post_axmodel"])
    caches_k = [np.zeros((MAX_CONTEXT, hidden_size), dtype=ml_dtypes.bfloat16) for _ in layers]
    caches_v = [np.zeros((MAX_CONTEXT, hidden_size), dtype=ml_dtypes.bfloat16) for _ in layers]

    prompt_len = len(ids)
    last_hidden = None
    for start in range(0, prompt_len, PREFILL_TOKENS):
        valid = min(PREFILL_TOKENS, prompt_len - start)
        group = start // PREFILL_TOKENS + 1
        block = np.zeros((1, PREFILL_TOKENS, hidden_size), dtype=ml_dtypes.bfloat16)
        block[0, :valid] = hidden[start : start + valid]
        indices = np.zeros((3, PREFILL_TOKENS), dtype=np.uint32)
        indices[:, :valid] = np.arange(start, start + valid, dtype=np.uint32)
        mask = causal_mask(start, valid)

        for layer_index, layer in enumerate(layers):
            cache_len = max(1, start)
            outputs = layer.run(
                None,
                {
                    "K_cache": caches_k[layer_index][None, :cache_len],
                    "V_cache": caches_v[layer_index][None, :cache_len],
                    "indices": indices,
                    "input": block,
                    "mask": mask,
                },
                shape_group=group,
            )
            caches_k[layer_index][start : start + valid] = outputs[0][0, :valid]
            caches_v[layer_index][start : start + valid] = outputs[1][0, :valid]
            block = outputs[2]
        last_hidden = block[:, valid - 1 : valid]

    if last_hidden is None:
        raise RuntimeError("prompt 为空")
    logits = post.run(None, {"input": last_hidden})[0]
    next_token = int(np.argmax(logits[0, 0]))
    generated: list[int] = []

    for position in range(prompt_len, min(prompt_len + max_new_tokens, MAX_CONTEXT)):
        if next_token in (IM_END_ID, ENDOFTEXT_ID):
            break
        generated.append(next_token)
        block = np.asarray(embeddings[next_token]).reshape(1, 1, hidden_size)
        indices = np.asarray([[position]], dtype=np.uint32)
        mask = np.full((1, 1, MAX_CONTEXT + 1), -65536, dtype=ml_dtypes.bfloat16)
        mask[0, 0, :position] = 0
        mask[0, 0, -1] = 0

        for layer_index, layer in enumerate(layers):
            outputs = layer.run(
                None,
                {
                    "K_cache": caches_k[layer_index][None],
                    "V_cache": caches_v[layer_index][None],
                    "indices": indices,
                    "input": block,
                    "mask": mask,
                },
                shape_group=0,
            )
            caches_k[layer_index][position] = outputs[0][0, 0]
            caches_v[layer_index][position] = outputs[1][0, 0]
            block = outputs[2]
        logits = post.run(None, {"input": block})[0]
        next_token = int(np.argmax(logits[0, 0]))

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    print(transcribe(args.model.resolve(), args.audio.resolve(), args.max_new_tokens))


if __name__ == "__main__":
    main()
