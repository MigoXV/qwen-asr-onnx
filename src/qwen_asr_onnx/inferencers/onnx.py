# coding=utf-8
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort

from qwen_asr_onnx.configs.constants import (
    QUANTIZE_INT8,
    TARGET_SAMPLE_RATE,
)

SAMPLE_RATE = TARGET_SAMPLE_RATE
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100

AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
NEWLINE_ID = 198

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024


def compute_mel_spectrogram(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    import librosa

    stft = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2
    mel_spec = mel_filters @ magnitudes

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def get_mel_filters() -> np.ndarray:
    import librosa

    mel_filters = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE // 2,
        norm="slaney",
        htk=False,
    )
    return mel_filters.astype(np.float32)


def get_feat_extract_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths


class SimpleTokenizer:
    """Tokenizer backed only by a local tokenizer.json file."""

    def __init__(self, tokenizer_path: Path) -> None:
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class OnnxAsrPipeline:
    """Pure ONNX Qwen ASR pipeline adapted for gRPC audio bytes input."""

    def __init__(
        self,
        model_root: Path,
        onnx_dir: Path,
        tokenizer_path: Path,
        num_threads: int = 0,
        quantize: str = QUANTIZE_INT8,
        max_new_tokens: int = 4096,
    ) -> None:
        self.model_root = Path(model_root)
        self.onnx_dir = Path(onnx_dir)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_new_tokens = max_new_tokens

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
        sess_opts.log_severity_level = 3

        decoder_init_path, decoder_step_path = self._resolve_decoder_paths(quantize)

        self.encoder_conv = ort.InferenceSession(
            str(self.onnx_dir / "encoder_conv.onnx"),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.encoder_transformer = ort.InferenceSession(
            str(self.onnx_dir / "encoder_transformer.onnx"),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_init = ort.InferenceSession(
            str(decoder_init_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_step = ort.InferenceSession(
            str(decoder_step_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

        embed_path = self.onnx_dir / "embed_tokens.bin"
        if not embed_path.is_file():
            raise FileNotFoundError(f"Embedding weights not found: {embed_path}")
        self.embed_tokens = np.fromfile(str(embed_path), dtype=np.float32).reshape(
            VOCAB_SIZE, HIDDEN_SIZE
        )

        self.mel_filters = get_mel_filters()
        self.tokenizer = SimpleTokenizer(self.tokenizer_path)

    def _resolve_decoder_paths(self, quantize: str) -> tuple[Path, Path]:
        if quantize == QUANTIZE_INT8:
            int8_init = self.onnx_dir / "decoder_init.int8.onnx"
            int8_step = self.onnx_dir / "decoder_step.int8.onnx"
            if int8_init.is_file() and int8_step.is_file():
                return int8_init, int8_step

        init_path = self.onnx_dir / "decoder_init.onnx"
        step_path = self.onnx_dir / "decoder_step.onnx"
        if not init_path.is_file() or not step_path.is_file():
            raise FileNotFoundError(
                "Decoder ONNX files not found. Expected either INT8 or FP32 decoder files "
                f"under {self.onnx_dir}."
            )
        return init_path, step_path

    @staticmethod
    def _audio_bytes_to_waveform(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        if sample_rate <= 0:
            raise ValueError("sample_rate_hertz must be > 0.")
        if not audio_bytes:
            raise ValueError("audio_content is empty.")
        if len(audio_bytes) % 2 != 0:
            raise ValueError("audio_content must contain 16-bit PCM bytes.")

        wav = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        return wav

    @staticmethod
    def _maybe_resample(wav: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == SAMPLE_RATE:
            return wav

        import librosa

        resampled = librosa.resample(
            wav,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )
        return resampled.astype(np.float32)

    def _prepare_waveform(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        wav = self._audio_bytes_to_waveform(audio_bytes, sample_rate)
        wav = self._maybe_resample(wav, sample_rate)
        if wav.ndim != 1:
            wav = np.reshape(wav, (-1,))
        return wav.astype(np.float32, copy=False)

    def _encode_audio(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        mel_valid = mel[:, :mel_len]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))
        chunk_lengths: list[int] = []

        for i in range(chunk_num):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, mel_len)
            chunk_lengths.append(end - start)

        max_chunk_len = max(chunk_lengths)
        padded = np.zeros((chunk_num, 1, N_MELS, max_chunk_len), dtype=np.float32)
        start = 0
        for i, chunk_length in enumerate(chunk_lengths):
            padded[i, 0, :, :chunk_length] = mel_valid[:, start : start + chunk_length]
            start += chunk_length

        lens_after_cnn = get_feat_extract_output_lengths(np.array(chunk_lengths))
        conv_out = self.encoder_conv.run(None, {"padded_mel_chunks": padded})[0]

        features = [conv_out[i, :length, :] for i, length in enumerate(lens_after_cnn)]
        hidden_states = np.concatenate(features, axis=0)

        total_tokens = hidden_states.shape[0]
        attn_mask = np.zeros((1, 1, total_tokens, total_tokens), dtype=np.float32)
        return self.encoder_transformer.run(
            None,
            {
                "hidden_states": hidden_states,
                "attention_mask": attn_mask,
            },
        )[0]

    def _build_prompt_ids(
        self,
        num_audio_tokens: int,
        language: str | None = None,
        context: str = "",
    ) -> list[int]:
        ids = [IM_START_ID] + self.tokenizer.encode("system") + [NEWLINE_ID]
        # Keep the interface for context/hotwords, but do not force prompt changes when empty.
        if context:
            ids += self.tokenizer.encode(context.strip()) + [NEWLINE_ID]
        ids += [IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.tokenizer.encode("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio_tokens + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]

        ids += [IM_START_ID] + self.tokenizer.encode("assistant") + [NEWLINE_ID]
        if language:
            ids += self.tokenizer.encode(f"language {language}<asr_text>")
        return ids

    def _embed_and_fuse(self, token_ids: list[int], audio_features: np.ndarray) -> np.ndarray:
        ids_array = np.array(token_ids)
        embeds = self.embed_tokens[ids_array]

        audio_positions = np.where(ids_array == AUDIO_PAD_ID)[0]
        if len(audio_positions) != audio_features.shape[0]:
            raise ValueError(
                "Audio token count mismatch: "
                f"{len(audio_positions)} vs {audio_features.shape[0]}"
            )
        embeds[audio_positions] = audio_features
        return embeds[np.newaxis, :, :]

    def _decode_tokens(
        self,
        input_embeds: np.ndarray,
        max_new_tokens: int,
    ) -> Iterator[str]:
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, present_keys, present_values = self.decoder_init.run(
            None,
            {
                "input_embeds": input_embeds,
                "position_ids": position_ids,
            },
        )

        generated: list[int] = []
        emitted_text = ""
        cur_pos = seq_len

        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            generated.append(next_token)
            raw_text = self.tokenizer.decode(generated)
            new_text = raw_text[len(emitted_text) :]
            for char in new_text:
                yield char
            emitted_text = raw_text

            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            pos = np.array([[cur_pos]], dtype=np.int64)
            logits, present_keys, present_values = self.decoder_step.run(
                None,
                {
                    "input_embeds": token_embed,
                    "position_ids": pos,
                    "past_keys": present_keys,
                    "past_values": present_values,
                },
            )
            cur_pos += 1

    def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language: str | None = None,
        context: str = "",
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        wav = self._prepare_waveform(audio_bytes, sample_rate)
        mel = compute_mel_spectrogram(wav, self.mel_filters)
        mel_len = mel.shape[1]
        audio_features = self._encode_audio(mel, mel_len)
        prompt_ids = self._build_prompt_ids(
            num_audio_tokens=audio_features.shape[0],
            language=language,
            context=context,
        )
        input_embeds = self._embed_and_fuse(prompt_ids, audio_features)
        decode_limit = max_new_tokens or self.max_new_tokens
        yield from self._decode_tokens(input_embeds, decode_limit)

    def close(self) -> None:
        return None
