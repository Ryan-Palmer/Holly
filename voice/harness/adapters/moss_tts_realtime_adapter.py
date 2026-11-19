"""MOSS-TTS-Realtime (OpenMOSS), 1.7B, 24 kHz. Streams natively, taking text
a word at a time the way an LLM produces it, which is what Holly would do.

Runs from the cloned repo (voice/harness/envs/moss-src), following its
example_llm_stream_to_tts.py. The reference is read with the standard
library, since the repo's loader goes through torchcodec, which needs
FFmpeg's DLLs on Windows.
"""

import sys
import wave
from pathlib import Path

import numpy as np

from common import Adapter

REPO = Path(__file__).resolve().parents[1] / "envs" / "moss-src" / "moss_tts_realtime"
MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
RATE = 24_000


class MossTtsRealtime(Adapter):
    name = "moss-tts-realtime"
    streams = True
    sample_rate = RATE

    def load(self):
        sys.path.insert(0, str(REPO))
        import torch
        from example_llm_stream_to_tts import decode_audio_frames, flush_decoder
        from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
        from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
        from mossttsrealtime.streaming_mossttsrealtime import (
            AudioStreamDecoder,
            MossTTSRealtimeInference,
            MossTTSRealtimeStreamingSession,
        )
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.decode_audio_frames, self.flush_decoder = decode_audio_frames, flush_decoder
        self.AudioStreamDecoder = AudioStreamDecoder
        self.device = torch.device("cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        processor = MossTTSRealtimeProcessor(tokenizer)
        model = MossTTSRealtime.from_pretrained(MODEL, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        model = model.to(self.device).eval()
        self.codec = AutoModel.from_pretrained(CODEC, trust_remote_code=True).eval().to(self.device)

        with wave.open(str(self.ref)) as w:
            assert w.getframerate() == RATE, "reference.wav should be 24 kHz (voice/prep.py export)"
            ref = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        with torch.inference_mode():
            enc = self.codec.encode(torch.from_numpy(ref)[None, None].to(self.device), chunk_duration=0.24)
            prompt_tokens = enc["audio_codes"].cpu().numpy().squeeze(1)

        inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=3000)
        # Without flash-attention (never on Windows) MOSS torch.compiles its
        # local transformer. Even with triton-windows installed, that never
        # got through the warm-up sentence: it kept recompiling. Run it eager.
        inferencer._should_compile_local_transformer = False
        inferencer.reset_generation_state(keep_cache=False)
        self.session = MossTTSRealtimeStreamingSession(
            inferencer, processor, codec=self.codec, codec_sample_rate=RATE,
            codec_encode_kwargs={"chunk_duration": 0.24}, prefill_text_len=processor.delay_tokens_len,
            temperature=0.8, top_p=0.6, top_k=30, do_sample=True,
            repetition_penalty=1.1, repetition_window=50,
        )
        self.session.set_voice_prompt_tokens(prompt_tokens)

        system_prompt = processor.make_ensemble(prompt_tokens)
        prefix_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
        prefix = np.full((len(prefix_ids), system_prompt.shape[1]), processor.audio_channel_pad, dtype=np.int64)
        prefix[:, 0] = prefix_ids
        self.input_ids = np.concatenate([system_prompt, prefix], axis=0)
        self.codebook_size = int(getattr(self.codec, "codebook_size", 1024))
        self.audio_eos = int(getattr(inferencer, "audio_eos_token", 1026))

    def synthesize(self, text):
        torch, s = self.torch, self.session
        s.reset_turn(input_ids=self.input_ids, include_system_prompt=False, reset_cache=True)
        decoder = self.AudioStreamDecoder(
            self.codec, chunk_frames=3, overlap_frames=0, decode_kwargs={"chunk_duration": -1}, device=self.device
        )
        frames = lambda fs: self.decode_audio_frames(fs, decoder, self.codebook_size, self.audio_eos)  # noqa: E731
        with torch.inference_mode(), self.codec.streaming(batch_size=1):
            # Word by word, as an LLM streams it.
            for i, word in enumerate(text.split(" ")):
                yield from frames(s.push_text(word if i == 0 else " " + word))
            yield from frames(s.end_text())
            while True:
                fs = s.drain(max_steps=1)
                if not fs:
                    break
                yield from frames(fs)
                if s.inferencer.is_finished:
                    break
            yield from self.flush_decoder(decoder)

    def version(self):
        return f"MOSS-TTS repo, {MODEL}"


if __name__ == "__main__":
    MossTtsRealtime.main()
