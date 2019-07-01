#!/usr/bin/env python
# coding: utf-8

# import matplotlib
# import matplotlib.pylab as plt

# import IPython.display as ipd

import sys
import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
# from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
# from denoiser import Denoiser
import os
import soundfile as sf
import pyaudio
import klepto
import IPython.display as ipd
import time
from sia.file_utils import cached_model_path

sys.path.append('waveglow/')


class TTSModel(object):
    """docstring for TTSModel."""

    def __init__(self):
        super(TTSModel, self).__init__()
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        self.model = load_model(hparams)
        tacotron2_path = cached_model_path("tacotron2_model")
        self.model.load_state_dict(
            torch.load(tacotron2_path, map_location='cpu')['state_dict'])
        self.model.eval()
        waveglow_path = cached_model_path('waveglow_model')
        self.waveglow = torch.load(waveglow_path, map_location='cpu')['model']
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

    def synth_speech(self, t):
        start = time.time()
        text = t
        sequence = np.array(text_to_sequence(text,
                                             ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(
            sequence)
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        # import ipdb; ipdb.set_trace()
        data = convert(audio[0].data.cpu().numpy())
        # _audio_stream.write(data.astype('float32'))
        # _audio_stream.write(data)
        end = time.time()
        print(end - start)
        return data.tobytes()


def convert(array):
    sf.write('sample.wav', array, 22050)
    os.system('ffmpeg -i {0} -filter:a "atempo=0.80" -ar 16k {1}'.format(
        'sample.wav', 'sample0.wav'))
    data, rate = sf.read('sample0.wav', dtype='int16')
    os.remove('sample.wav')
    os.remove('sample0.wav')
    return data


def display(data):
    aud = ipd.Audio(data, rate=16000)
    return aud


def player_gen():
    audio_interface = pyaudio.PyAudio()
    _audio_stream = audio_interface.open(format=pyaudio.paInt16,
                                         channels=1,
                                         rate=16000,
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