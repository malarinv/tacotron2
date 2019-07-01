# -*- coding: utf-8 -*-
import grpc
import time
from sia.proto import tts_pb2
from sia.proto import tts_pb2_grpc
from concurrent import futures
from sia.instruments import do_time
from tts import TTSModel


class TTSServer():
    def __init__(self):
        self.tts_model = TTSModel()

    def TextToSpeechAPI(self, request, context):
        while (True):
            input_text = request.text
            speech_response = self.tts_model.synth_speech(input_text)
            return tts_pb2.SpeechResponse(response=speech_response)


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tts_server = TTSServer()
    tts_pb2_grpc.add_ServerServicer_to_server(tts_server, server)
    server.add_insecure_port('localhost:50060')
    server.start()
    print('TTSServer started!')

    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.start()
        # server.stop(0)


if __name__ == "__main__":
    main()
