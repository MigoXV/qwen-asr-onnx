#!/usr/bin/env python3
"""
Qwen3-ASR-0.6B — Pure ONNX Inference Pipeline.
No PyTorch dependency. Uses only ONNX Runtime + NumPy + librosa.
Architecture:
    Audio → Mel → Encoder (ONNX) → Audio Features
    Prompt tokens → Embed (numpy) → Replace audio placeholders → Decoder Init (ONNX) → Logits + KV Cache
    Greedy decode loop: Decoder Step (ONNX) → next token until EOS
Usage:
    python onnx_inference.py audio.wav
    python onnx_inference.py audio1.wav audio2.wav --language Korean
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import onnxruntime as ort


# ── Constants ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100  # n_window * 2

# Special token IDs
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645      # EOS
ENDOFTEXT_ID = 151643   # EOS alt
NEWLINE_ID = 198        # '\n'

# Vocab
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024


# ── Mel Spectrogram (Whisper-compatible, no PyTorch) ────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio file as mono 16kHz float32."""
    import librosa
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def compute_mel_spectrogram(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram (Whisper-compatible).
    Args:
        wav: [T] float32 audio
        mel_filters: [n_mels, n_fft//2+1] mel filterbank
    Returns:
        mel: [n_mels, n_frames] float32
    """
    import librosa

    # STFT
    stft = librosa.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window="hann", center=True, pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2

    # Apply mel filterbank
    mel_spec = mel_filters @ magnitudes

    # Log scale (Whisper-style)
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)


def get_mel_filters() -> np.ndarray:
    """Get Whisper-compatible mel filterbank using librosa."""
    import librosa
    mel_filters = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        fmin=0, fmax=SAMPLE_RATE // 2, norm="slaney", htk=False,
    )
    return mel_filters.astype(np.float32)


def get_feat_extract_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    """Compute output lengths after 3x stride-2 convolution."""
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths


# ── VAD-based Long Audio Chunking ────────────────────────────────────

SILENCE_THRESHOLD_DB = -40
SILENCE_HOP_SEC = 0.1


def find_silence_split_points(wav: np.ndarray, target_sec: int = 30) -> list:
    """Find sample indices where audio can be split at silence boundaries.
    Uses RMS energy to detect silence — no external VAD model needed.
    Splits long audio into chunks at the nearest silent frame.
    Args:
        wav: Audio samples (float32, 16kHz mono).
        target_sec: Target chunk length in seconds (default 30).
                     Min = target/2, Max = target*1.5.
    """
    import librosa

    min_sec = target_sec // 2
    max_sec = int(target_sec * 1.5)

    total_samples = len(wav)
    if total_samples <= max_sec * SAMPLE_RATE:
        return []

    hop_samples = int(SILENCE_HOP_SEC * SAMPLE_RATE)
    rms = librosa.feature.rms(
        y=wav, frame_length=hop_samples * 2, hop_length=hop_samples,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_silent = rms_db < SILENCE_THRESHOLD_DB

    split_points = []
    cursor = 0

    while cursor + max_sec * SAMPLE_RATE < total_samples:
        search_start_sec = max(0, cursor / SAMPLE_RATE + min_sec)
        search_end_sec = cursor / SAMPLE_RATE + max_sec
        target_abs_sec = cursor / SAMPLE_RATE + target_sec

        frame_start = int(search_start_sec / SILENCE_HOP_SEC)
        frame_end = min(int(search_end_sec / SILENCE_HOP_SEC), len(is_silent))
        frame_target = int(target_abs_sec / SILENCE_HOP_SEC)

        silent_frames = np.where(is_silent[frame_start:frame_end])[0] + frame_start

        if len(silent_frames) > 0:
            best_idx = int(np.argmin(np.abs(silent_frames - frame_target)))
            split_frame = silent_frames[best_idx]
            split_sample = int(split_frame * hop_samples)
        else:
            split_sample = int(target_abs_sec * SAMPLE_RATE)

        split_sample = min(split_sample, total_samples)
        split_points.append(split_sample)
        cursor = split_sample

    return split_points


# ── Tokenizer (minimal, no HuggingFace dependency) ─────────────────────

class SimpleTokenizer:
    """Minimal tokenizer using tokenizers library (or HF tokenizer.json)."""

    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path and Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Fall back to HF tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
            self._is_hf = True
            return
        self._is_hf = False

    def encode(self, text: str) -> list:
        if self._is_hf:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        if self._is_hf:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# ── ONNX Pipeline ──────────────────────────────────────────────────────

class OnnxAsrPipeline:
    """End-to-end ASR pipeline using only ONNX Runtime."""

    def __init__(self, onnx_dir: str = "onnx_models", num_threads: int = 0,
                 quantize: str = "int8"):
        onnx_path = Path(onnx_dir)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
        sess_opts.log_severity_level = 3  # Suppress warnings

        # Choose decoder model files based on quantization
        if quantize == "int8" and (onnx_path / "decoder_init.int8.onnx").exists():
            decoder_init_path = "decoder_init.int8.onnx"
            decoder_step_path = "decoder_step.int8.onnx"
            print(f"Loading ONNX models (decoder: INT8)...")
        else:
            decoder_init_path = "decoder_init.onnx"
            decoder_step_path = "decoder_step.onnx"
            print(f"Loading ONNX models (decoder: FP32)...")

        self.encoder_conv = ort.InferenceSession(
            str(onnx_path / "encoder_conv.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.encoder_transformer = ort.InferenceSession(
            str(onnx_path / "encoder_transformer.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_init = ort.InferenceSession(
            str(onnx_path / decoder_init_path), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_step = ort.InferenceSession(
            str(onnx_path / decoder_step_path), sess_opts,
            providers=["CPUExecutionProvider"])

        # Load embedding matrix
        embed_path = onnx_path / "embed_tokens.bin"
        print(f"Loading embeddings ({embed_path.stat().st_size / 1e6:.0f} MB)...")
        self.embed_tokens = np.fromfile(
            str(embed_path), dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)

        # Mel filterbank
        self.mel_filters = get_mel_filters()

        # Tokenizer
        tokenizer_path = onnx_path / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = None
        self.tokenizer = SimpleTokenizer(str(tokenizer_path) if tokenizer_path else None)

        print("Pipeline ready.")

    def _encode_audio(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        """Run encoder: mel → audio features [N, 1024]."""
        mel_valid = mel[:, :mel_len]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))

        chunk_lengths = []
        for i in range(chunk_num):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, mel_len)
            chunk_lengths.append(end - start)

        # Pad chunks
        max_chunk_len = max(chunk_lengths)
        padded = np.zeros((chunk_num, 1, N_MELS, max_chunk_len), dtype=np.float32)
        start = 0
        for i, cl in enumerate(chunk_lengths):
            padded[i, 0, :, :cl] = mel_valid[:, start:start + cl]
            start += cl

        # Conv output lengths
        lens_after_cnn = get_feat_extract_output_lengths(np.array(chunk_lengths))

        # Conv block
        conv_out = self.encoder_conv.run(None, {"padded_mel_chunks": padded})[0]

        # Pack features (remove padding)
        features = []
        for i, l in enumerate(lens_after_cnn):
            features.append(conv_out[i, :l, :])
        hidden_states = np.concatenate(features, axis=0)

        # Transformer block (all-to-all attention)
        total_tokens = hidden_states.shape[0]
        attn_mask = np.zeros((1, 1, total_tokens, total_tokens), dtype=np.float32)
        encoder_output = self.encoder_transformer.run(None, {
            "hidden_states": hidden_states,
            "attention_mask": attn_mask,
        })[0]

        return encoder_output  # [N, 1024]

    def _build_prompt_ids(self, num_audio_tokens: int, language: Optional[str] = None) -> list:
        """Build prompt token IDs with audio placeholders."""
        # <|im_start|>system\n<|im_end|>\n
        ids = [IM_START_ID] + self.tokenizer.encode("system") + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]
        # <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
        ids += [IM_START_ID] + self.tokenizer.encode("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio_tokens + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]
        # <|im_start|>assistant\n
        ids += [IM_START_ID] + self.tokenizer.encode("assistant") + [NEWLINE_ID]
        if language:
            lang_tokens = self.tokenizer.encode(f"language {language}<asr_text>")
            ids += lang_tokens
        return ids

    def _embed_and_fuse(self, token_ids: list, audio_features: np.ndarray) -> np.ndarray:
        """Embed tokens and replace audio placeholders with encoder output."""
        ids_array = np.array(token_ids)
        embeds = self.embed_tokens[ids_array]  # [seq_len, 1024]

        # Replace audio_pad positions
        audio_mask = (ids_array == AUDIO_PAD_ID)
        audio_positions = np.where(audio_mask)[0]
        assert len(audio_positions) == audio_features.shape[0], \
            f"Audio token count mismatch: {len(audio_positions)} vs {audio_features.shape[0]}"
        embeds[audio_positions] = audio_features

        return embeds[np.newaxis, :, :]  # [1, seq_len, 1024]

    def _transcribe_chunk(
        self,
        wav: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Iterator[str]:
        """Transcribe a single audio chunk and yield decoded text incrementally."""
        mel = compute_mel_spectrogram(wav, self.mel_filters)
        mel_len = mel.shape[1]

        audio_features = self._encode_audio(mel, mel_len)
        num_audio_tokens = audio_features.shape[0]

        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        input_embeds = self._embed_and_fuse(token_ids, audio_features)
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, present_keys, present_values = self.decoder_init.run(None, {
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })

        generated: list[int] = []
        emitted_text = ""
        cur_pos = seq_len

        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            generated.append(next_token)
            raw_text = self.tokenizer.decode(generated)
            if "language " in raw_text and "<asr_text>" in raw_text:
                text = raw_text.split("<asr_text>", 1)[1]
            elif raw_text.startswith("language "):
                text = ""
            else:
                text = raw_text

            new_text = text[len(emitted_text):]
            for char in new_text:
                yield char
            emitted_text = text

            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            pos = np.array([[cur_pos]], dtype=np.int64)

            logits, present_keys, present_values = self.decoder_step.run(None, {
                "input_embeds": token_embed,
                "position_ids": pos,
                "past_keys": present_keys,
                "past_values": present_values,
            })

            cur_pos += 1

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
        chunk_sec: int = 30,
    ) -> Iterator[str]:
        """Transcribe an audio file and yield decoded text incrementally."""
        wav = load_audio(audio_path)
        yield from self._transcribe_chunk(wav, language, max_new_tokens)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Pure ONNX Inference")
    parser.add_argument("--audio", default="data-bin/zh.mp3")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--onnx-dir", type=str, default="model-bin/Daumee/Qwen3-ASR-0.6B-ONNX-CPU/onnx_models")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quantize", type=str, default="int8", choices=["none", "int8"],
                        help="Decoder quantization: none (FP32) or int8 (default)")
    parser.add_argument("--chunk-sec", type=int, default=30,
                        help="Target chunk length for long audio splitting (default: 30)")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads (0=all)")
    args = parser.parse_args()

    pipeline = OnnxAsrPipeline(onnx_dir=args.onnx_dir, num_threads=args.threads,
                               quantize=args.quantize)

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}", file=sys.stderr)

    text_stream = pipeline.transcribe(
        audio_path, language=args.language,
        max_new_tokens=args.max_new_tokens,
        chunk_sec=args.chunk_sec,
    )

    for char in text_stream:
        print(char, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
