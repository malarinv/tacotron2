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
hparams = create_hparams()
hparams.sampling_rate = 22050
model = load_model(hparams)
tacotron2_path = cached_model_path("tacotron2_model")
model.load_state_dict(
    torch.load(tacotron2_path, map_location='cpu')['state_dict'])
model.eval()
waveglow_path = cached_model_path('waveglow_model')
waveglow = torch.load(waveglow_path, map_location='cpu')['model']
waveglow.eval()
for k in waveglow.convinv:
    k.float()
k_cache = klepto.archives.file_archive(cached=False)

# https://github.com/NVIDIA/waveglow/issues/127
for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')


def convert(array):
    sf.write('sample.wav', array, 22050)
    os.system('ffmpeg -i {0} -filter:a "atempo=0.80" -ar 16k {1}'.format(
        'sample.wav', 'sample0.wav'))
    data, rate = sf.read('sample0.wav', dtype='int16')
    os.remove('sample.wav')
    os.remove('sample0.wav')
    return data


@klepto.safe.inf_cache(cache=k_cache)
def speech(t):
    start = time.time()
    text = t
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    # import ipdb; ipdb.set_trace()
    data = convert(audio[0].data.cpu().numpy())
    # _audio_stream.write(data.astype('float32'))
    # _audio_stream.write(data)
    end = time.time()
    print(end - start)
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
        _audio_stream.write(data.tostring())
        # _audio_stream.close()

    return play_device


def synthesize_corpus():
    all_data = []
    for (i, line) in enumerate(open('corpus.txt').readlines()):
        print('synthesizing... "{}"'.format(line.strip()))
        data = speech(line.strip())
        sf.write('tts_{}.wav'.format(i), data, 16000)
        all_data.append(data)
    return all_data


def play_corpus(corpus_synths):
    player = player_gen()
    for d in corpus_synths:
        player(d)


def main():
    # data = speech('Hi I am Sia. How may I help you today .'.lower())
    # audio_interface = pyaudio.PyAudio()
    # _audio_stream = audio_interface.open(format=pyaudio.paInt16,
    #                                      channels=1,
    #                                      rate=16000,
    #                                      output=True)
    # _audio_stream.write(data)
    corpus_synth_data = synthesize_corpus()
    play_corpus(corpus_synth_data)
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
