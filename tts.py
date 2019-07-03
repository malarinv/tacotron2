#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from .hparams import create_hparams
from .text import text_to_sequence
from .glow import WaveGlow
# import os
# import soundfile as sf
import pyaudio
import klepto
from librosa import resample
from librosa.effects import time_stretch
from sia.file_utils import cached_model_path
from sia.instruments import do_time
from .model import Tacotron2

TTS_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 16000

# https://github.com/NVIDIA/waveglow/blob/master/config.json
WAVEGLOW_CONFIG = {
    "n_mel_channels": 80,
    "n_flows": 12,
    "n_group": 8,
    "n_early_every": 4,
    "n_early_size": 2,
    "WN_config": {
        "n_layers": 8,
        "n_channels": 256,
        "kernel_size": 3
    }
}


class TTSModel(object):
    """docstring for TTSModel."""

    def __init__(self):
        super(TTSModel, self).__init__()
        hparams = create_hparams()
        hparams.sampling_rate = TTS_SAMPLE_RATE
        self.model = Tacotron2(hparams)
        tacotron2_path = cached_model_path("tacotron2_model")
        self.model.load_state_dict(
            torch.load(tacotron2_path, map_location='cpu')['state_dict'])
        self.model.eval()
        waveglow_path = cached_model_path('waveglow_model')
        self.waveglow = WaveGlow(**WAVEGLOW_CONFIG)
        wave_params = torch.load(waveglow_path, map_location='cpu')
        self.waveglow.load_state_dict(wave_params)
        self.waveglow.eval()
        for k in self.waveglow.convinv:
            k.float()
        self.k_cache = klepto.archives.file_archive(cached=False)
        self.synth_speech = klepto.safe.inf_cache(cache=self.k_cache)(
            self.synth_speech)

        # https://github.com/NVIDIA/waveglow/issues/127
        for m in self.waveglow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

    @do_time
    def synth_speech(self, t):
        text = t
        sequence = np.array(text_to_sequence(text,
                                             ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(
            sequence)
        with torch.no_grad():
            audio_t = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio = audio_t[0].data.cpu().numpy()
        # data = convert(audio)
        slow_data = time_stretch(audio, 0.8)
        float_data = resample(slow_data, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        data = float2pcm(float_data)
        return data.tobytes()


# def convert(array):
#     sf.write('sample.wav', array, TTS_SAMPLE_RATE)
#     # convert to $OUTPUT_SAMPLE_RATE
#     os.system('ffmpeg -i {0} -filter:a "atempo=0.80" -ar 16k {1}'.format(
#         'sample.wav', 'sample0.wav'))
#     data, rate = sf.read('sample0.wav', dtype='int16')
#     os.remove('sample.wav')
#     os.remove('sample0.wav')
#     return data


# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def float2pcm(sig, dtype='int16'):
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
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2**(i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def display(data):
    import IPython.display as ipd
    aud = ipd.Audio(data, rate=16000)
    return aud


def player_gen():
    audio_interface = pyaudio.PyAudio()
    _audio_stream = audio_interface.open(format=pyaudio.paInt16,
                                         channels=1,
                                         rate=OUTPUT_SAMPLE_RATE,
                                         output=True)

    def play_device(data):
        _audio_stream.write(data)
        # _audio_stream.close()

    return play_device


def synthesize_corpus():
    tts_model = TTSModel()
    all_data = []
    for (i, line) in enumerate(open('corpus.txt').readlines()):
        print('synthesizing... "{}"'.format(line.strip()))
        data = tts_model.synth_speech(line.strip())
        all_data.append(data)
    return all_data


def play_corpus(corpus_synths):
    player = player_gen()
    for d in corpus_synths:
        player(d)


def main():
    corpus_synth_data = synthesize_corpus()
    play_corpus(corpus_synth_data)
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
