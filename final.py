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
import time

sys.path.append('waveglow/')
hparams = create_hparams()
hparams.sampling_rate = 22050
checkpoint_path = "checkpoint_15000"
model = load_model(hparams)
model.load_state_dict(
    torch.load(checkpoint_path, map_location='cpu')['state_dict'])
model.eval()
waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path, map_location='cpu')['model']
waveglow.eval()
for k in waveglow.convinv:
    k.float()

audio_interface = pyaudio.PyAudio()
# _audio_stream = audio_interface.open(format=pyaudio.paFloat32,channels=1, rate=22050,output=True)
_audio_stream = audio_interface.open(format=pyaudio.paInt16,channels=1, rate=16000,output=True)

# https://github.com/NVIDIA/waveglow/issues/127
for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')


def convert(array):
    sf.write('sample.wav', array, 22050)
    os.system('ffmpeg -i {0} -filter:a "atempo=0.80" -ar 16k {1}'.format('sample.wav', 'sample0.wav'))
    data, rate = sf.read('sample0.wav', dtype='int16')
    os.remove('sample.wav')
    os.remove('sample0.wav')
    return data


def speech(t):
    start = time.time()
    text = t
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    import ipdb; ipdb.set_trace()
    data = convert(audio[0].data.cpu().numpy())
    # _audio_stream.write(data.astype('float32'))
    _audio_stream.write(data)
    end = time.time()
    print(end - start)


def main():
    speech(
        ('I understand your frustration and disappointment. I am sorry that'
         ' its happening and I would like to help prevent it in the future. '
         'What style of diapers did you buy? For instance, was it the '
         'snugglers, pull ups or baby dry.'))


if __name__ == '__main__':
    main()
