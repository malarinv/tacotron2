import os

from google.cloud import texttospeech
from ..tts import TTSModel


tts_model_weights = os.environ.get(
    "TTS_MODELS", "models/tacotron2_statedict.pt,models/waveglow_256channels.pt"
)

tts_creds = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", "/code/config/gre2e/keys/gre2e_gcp.json"
)
taco2, wav_glow = tts_model_weights.split(",", 1)


class TTSSynthesizer(object):
    """docstring for TTSSynthesizer."""

    def __init__(self, backend="taco2"):
        super(TTSSynthesizer, self).__init__()
        if backend == "taco2":
            tts_model = TTSModel(f"{taco2}", f"{wav_glow}")  # Loads the models
            self.synth_speech = tts_model.synth_speech
        elif backend == "gcp":
            client = texttospeech.TextToSpeechClient()
            # Build the voice request, select the language code ("en-US") and the ssml
            # voice gender ("neutral")
            voice = texttospeech.types.VoiceSelectionParams(language_code="en-US")

            # Select the type of audio file you want returned
            audio_config = texttospeech.types.AudioConfig(
                audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16
            )

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            def gcp_synthesize(speech_text):
                synthesis_input = texttospeech.types.SynthesisInput(text=speech_text)
                response = client.synthesize_speech(
                    synthesis_input, voice, audio_config
                )
                return response.audio_content

            self.synth_speech = gcp_synthesize
