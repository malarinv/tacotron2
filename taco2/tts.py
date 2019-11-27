#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pyaudio
import klepto
import argparse
from pathlib import Path
from .model import Tacotron2
from glow import WaveGlow
from .hparams import HParams
from .layers import TacotronSTFT
from .text import text_to_sequence
from .denoiser import Denoiser
from .audio_processing import griffin_lim, postprocess_audio

TTS_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
# OUTPUT_SAMPLE_RATE = 16000

# config from
# https://github.com/NVIDIA/waveglow/blob/master/config.json
WAVEGLOW_CONFIG = {
    "n_mel_channels": 40,
    "n_flows": 12,
    "n_group": 8,
    "n_early_every": 4,
    "n_early_size": 2,
    "WN_config": {"n_layers": 8, "n_channels": 256, "kernel_size": 3},
}


class TTSModel(object):
    """docstring for TTSModel."""

    def __init__(self, tacotron2_path, waveglow_path, **kwargs):
        super(TTSModel, self).__init__()
        hparams = HParams(**kwargs)
        hparams.sampling_rate = TTS_SAMPLE_RATE
        self.model = Tacotron2(hparams)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(tacotron2_path)["state_dict"])
            self.model.cuda().eval()
        else:
            self.model.load_state_dict(
                torch.load(tacotron2_path, map_location="cpu")["state_dict"]
            )
            self.model.eval()
        self.k_cache = klepto.archives.file_archive(cached=False)
        if waveglow_path:
            if torch.cuda.is_available():
                wave_params = torch.load(waveglow_path)
            else:
                wave_params = torch.load(waveglow_path, map_location="cpu")
            try:
                self.waveglow = WaveGlow(**WAVEGLOW_CONFIG)
                self.waveglow.load_state_dict(wave_params)
            except:
                self.waveglow = wave_params["model"]
                self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
            if torch.cuda.is_available():
                self.waveglow.cuda().eval()
            else:
                self.waveglow.eval()
            # workaround from
            # https://github.com/NVIDIA/waveglow/issues/127
            for m in self.waveglow.modules():
                if "Conv" in str(type(m)):
                    setattr(m, "padding_mode", "zeros")
            for k in self.waveglow.convinv:
                k.float().half()
            self.denoiser = Denoiser(
                self.waveglow, n_mel_channels=hparams.n_mel_channels
            )
            self.synth_speech = klepto.safe.inf_cache(cache=self.k_cache)(
                self.synth_speech
            )
        else:
            self.synth_speech = klepto.safe.inf_cache(cache=self.k_cache)(
                self.synth_speech_gl
            )
        self.taco_stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            n_mel_channels=hparams.n_mel_channels,
            sampling_rate=hparams.sampling_rate,
            mel_fmax=4000,
        )

    def generate_mel_postnet(self, text):
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
        if torch.cuda.is_available():
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        else:
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(
                sequence
            )
        return mel_outputs_postnet

    def synth_speech_array(self, text):
        mel_outputs_postnet = self.generate_mel_postnet(text)

        with torch.no_grad():
            audio_t = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio_t = self.denoiser(audio_t, 0.1)[0]
        audio = audio_t[0].data.cpu().numpy()
        return audio

    def synth_speech(self, text):
        audio = self.synth_speech_array(text)

        return postprocess_audio(
            audio, src_rate=TTS_SAMPLE_RATE, dst_rate=OUTPUT_SAMPLE_RATE
        )

    def synth_speech_gl(self, text, griffin_iters=60):
        mel_outputs_postnet = self.generate_mel_postnet(text)

        mel_decompress = self.taco_stft.spectral_de_normalize(mel_outputs_postnet)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], self.taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        audio = griffin_lim(
            torch.autograd.Variable(spec_from_mel[:, :, :-1]),
            self.taco_stft.stft_fn,
            griffin_iters,
        )
        audio = audio.squeeze()
        audio = audio.cpu().numpy()

        return postprocess_audio(
            audio, tempo=0.6, src_rate=TTS_SAMPLE_RATE, dst_rate=OUTPUT_SAMPLE_RATE
        )


def player_gen():
    audio_interface = pyaudio.PyAudio()
    _audio_stream = audio_interface.open(
        format=pyaudio.paInt16, channels=1, rate=OUTPUT_SAMPLE_RATE, output=True
    )

    def play_device(data):
        _audio_stream.write(data)
        # _audio_stream.close()

    return play_device


def repl(tts_model):
    player = player_gen()

    def loop():
        text = input("tts >")
        data = tts_model.synth_speech(text.strip())
        player(data)

    return loop


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--tacotron2_path",
        type=Path,
        default="./tacotron.pt",
        help="Path to a tacotron2 model",
    )
    parser.add_argument(
        "-w",
        "--waveglow_path",
        type=Path,
        default="./waveglow_256channels.pt",
        help="Path to a waveglow model",
    )
    args = parser.parse_args()
    tts_model = TTSModel(**vars(args))
    interactive_loop = repl(tts_model)
    while True:
        interactive_loop()


if __name__ == "__main__":
    main()
