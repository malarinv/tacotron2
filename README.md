# Taco2 TTS

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> Generates speech audio from text
---

# Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

# Features

* TTS using Tacotron2


# Installation
To install the packages and its dependencies run.
```bash
python setup.py install
```
or with pip
```bash
pip install .
```

The installation should work on Python 3.6 or newer. Untested on Python 2.7

# Usage
```python
from taco2.tts import TTSModel
tts_model = TTSModel("/path/to/tacotron2_model","/path/to/waveglow_model") # Loads the models
SPEECH_AUDIO = tts_model.synth_speech(TEXT) # Returns the wav buffer
```
If `'/path/to/waveglow_model'` is `None` *Griffin-Lim vocoder* will be used.
