#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pyaudio
from librosa import resample
from librosa.effects import time_stretch
import klepto
from .model import Tacotron2
from glow import WaveGlow
from .hparams import HParams
from .text import text_to_sequence
from .denoiser import Denoiser

TTS_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 16000

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

    def __init__(self, tacotron2_path, waveglow_path):
        super(TTSModel, self).__init__()
        hparams = HParams()
        hparams.sampling_rate = TTS_SAMPLE_RATE
        self.model = Tacotron2(hparams)
        self.model.load_state_dict(
            torch.load(tacotron2_path, map_location="cpu")["state_dict"]
        )
        self.model.eval()
        wave_params = torch.load(waveglow_path, map_location="cpu")
        try:
            self.waveglow = WaveGlow(**WAVEGLOW_CONFIG)
            self.waveglow.load_state_dict(wave_params)
            self.waveglow.eval()
        except:
            self.waveglow = wave_params["model"]
            self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
            self.waveglow.eval()
        # workaround from
        # https://github.com/NVIDIA/waveglow/issues/127
        for m in self.waveglow.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")
        for k in self.waveglow.convinv:
            k.float()
        self.k_cache = klepto.archives.file_archive(cached=False)
        self.synth_speech = klepto.safe.inf_cache(cache=self.k_cache)(self.synth_speech)
        self.denoiser = Denoiser(self.waveglow)

    def synth_speech(self, text):
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
        # width = mel_outputs_postnet.shape[2]
        # wave_glow_input = torch.randn(1, 80, width)*0.00001
        # wave_glow_input[:,40:,:] = mel_outputs_postnet
        with torch.no_grad():
            audio_t = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio_t = self.denoiser(audio_t, 0.1)[0]
        audio = audio_t[0].data.cpu().numpy()
        # data = convert(audio)
        slow_data = time_stretch(audio, 0.8)
        float_data = resample(slow_data, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        data = float2pcm(float_data)
        return data.tobytes()

    def synth_speech_algo(self, text, griffin_iters=60):
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
        from .hparams import HParams
        from .layers import TacotronSTFT
        from .audio_processing import griffin_lim

        hparams = HParams()
        taco_stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            n_mel_channels=hparams.n_mel_channels,
            sampling_rate=hparams.sampling_rate,
            mel_fmax=4000,
        )
        mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        audio = griffin_lim(
            torch.autograd.Variable(spec_from_mel[:, :, :-1]),
            taco_stft.stft_fn,
            griffin_iters,
        )
        audio = audio.squeeze()
        audio = audio.cpu().numpy()

        slow_data = time_stretch(audio, 0.8)
        float_data = resample(slow_data, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        data = float2pcm(float_data)
        return data.tobytes()


# adapted from
# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def player_gen():
    audio_interface = pyaudio.PyAudio()
    _audio_stream = audio_interface.open(
        format=pyaudio.paInt16, channels=1, rate=OUTPUT_SAMPLE_RATE, output=True
    )

    def play_device(data):
        _audio_stream.write(data)
        # _audio_stream.close()

    return play_device


def repl():
    tts_model = TTSModel("/path/to/tacotron2.pt", "/path/to/waveglow.pt")
    player = player_gen()

    def loop():
        text = input("tts >")
        data = tts_model.synth_speech(text.strip())
        player(data)

    return loop


def main():
    interactive_loop = repl()
    while True:
        interactive_loop()


if __name__ == "__main__":
    main()
