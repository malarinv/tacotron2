# -*- coding: utf-8 -*-
import grpc
from sia.proto import tts_pb2
from sia.proto import tts_pb2_grpc
from .tts import player_gen


def tts_player():
    player = player_gen()
    channel = grpc.insecure_channel('localhost:50060')
    stub = tts_pb2_grpc.ServerStub(channel)

    def play(t):
        test_text = tts_pb2.TextInput(text=t)
        speech = stub.TextToSpeechAPI(test_text)
        player(speech.response)

    return play


def main():
    play = tts_player()
    play('How may I help you today?')
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
