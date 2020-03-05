import os
import logging

import rpyc
from rpyc.utils.server import ThreadedServer

from .backend import TTSSynthesizer


tts_backend = os.environ.get("TTS_BACKEND", "taco2")
tts_synthesizer = TTSSynthesizer(backend=tts_backend)


class TTSService(rpyc.Service):
    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_synth_speech(self, utterance: str):  # this is an exposed method
        speech_audio = tts_synthesizer.synth_speech(utterance)
        return speech_audio

    def exposed_synth_speech_cb(
        self, utterance: str, respond
    ):  # this is an exposed method
        speech_audio = tts_synthesizer.synth_speech(utterance)
        respond(speech_audio)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    port = int(os.environ.get("TTS_RPYC_PORT", "7754"))
    logging.info("starting tts server...")
    t = ThreadedServer(TTSService, port=port)
    t.start()


if __name__ == "__main__":
    main()
