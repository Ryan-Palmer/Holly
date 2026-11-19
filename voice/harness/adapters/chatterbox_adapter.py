"""Chatterbox (Resemble AI), the original 500M model. No reference transcript
needed. Output carries Resemble's inaudible PerTh watermark."""

from importlib.metadata import version

from common import Adapter


class Chatterbox(Adapter):
    name = "chatterbox"

    def load(self):
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.sample_rate = self.model.sr

    def synthesize(self, text):
        yield self.model.generate(text, audio_prompt_path=str(self.ref))

    def version(self):
        return version("chatterbox-tts")


if __name__ == "__main__":
    Chatterbox.main()
