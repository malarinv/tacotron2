import grpc
from sia.proto import tts_pb2
from sia.proto import tts_pb2_grpc
from tts import player_gen


def main():
    channel = grpc.insecure_channel('localhost:50060')
    stub = tts_pb2_grpc.ServerStub(channel)
    test_text = tts_pb2.TextInput(text='How may I help you today?')
    speech = stub.TextToSpeechAPI(test_text)
    player = player_gen()
    player(speech.response)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
