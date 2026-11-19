"""Chatterbox Turbo (Resemble AI): 350M, English only, faster than the
original. Shares the chatterbox venv."""

from importlib.metadata import version

from common import Adapter


class ChatterboxTurbo(Adapter):
    name = "chatterbox-turbo"

    def load(self):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.sample_rate = self.model.sr

    def synthesize(self, text):
        yield self.model.generate(text, audio_prompt_path=str(self.ref))

    def version(self):
        return version("chatterbox-tts") + " (turbo)"


if __name__ == "__main__":
    ChatterboxTurbo.main()
