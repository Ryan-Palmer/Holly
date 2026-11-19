"""Qwen3-TTS 1.7B Base (Alibaba). Needs the reference transcript; the
reference is encoded once into a reusable voice-clone prompt. The official
package has no streaming call, so this times whole sentences."""

from importlib.metadata import version

from common import Adapter

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class Qwen3Tts(Adapter):
    name = "qwen3-tts"

    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        # No flash-attn on Windows; transformers falls back to SDPA.
        self.model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(self.ref), ref_text=self.ref_text)

    def synthesize(self, text):
        wavs, sr = self.model.generate_voice_clone(text=text, language="English", voice_clone_prompt=self.prompt)
        self.sample_rate = sr
        yield wavs[0]

    def version(self):
        return f"qwen-tts {version('qwen-tts')}, {MODEL}"


if __name__ == "__main__":
    Qwen3Tts.main()
