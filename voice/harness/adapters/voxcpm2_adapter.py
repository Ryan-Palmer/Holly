"""VoxCPM2 (OpenBMB), 2B, 48 kHz. Streams. Uses both of its cloning inputs:
the isolated reference audio, and the reference as a prompt with its
transcript ("continuation" mode), which its README calls highest fidelity."""

from importlib.metadata import version

from common import Adapter

MODEL = "openbmb/VoxCPM2"


class VoxCpm2(Adapter):
    name = "voxcpm2"
    streams = True

    def load(self):
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(MODEL, load_denoiser=False, device="cuda")
        tts = getattr(self.model, "tts_model", self.model)
        self.sample_rate = int(getattr(tts, "sample_rate", 48_000))

    def synthesize(self, text):
        yield from self.model.generate_streaming(
            text=text,
            reference_wav_path=str(self.ref),
            prompt_wav_path=str(self.ref),
            prompt_text=self.ref_text,
        )

    def version(self):
        return f"voxcpm {version('voxcpm')}, {MODEL}"


if __name__ == "__main__":
    VoxCpm2.main()
