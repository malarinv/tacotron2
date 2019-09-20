# Taco2 TTS

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> Generate speech audio from text
---

# Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

# Features

* Tacotron2 Synthesized Speech


# Installation
Install the packages with for production use. It downloads the dependencies
```bash
python setup.py install
```

> Still facing an issue? Check the [Issues](#issues) section or open a new issue.

The installation should be smooth with Python 3.6 or newer.

# Usage
> API
```python
tts_model = TTSModel("/path/to/tacotron2_model","/path/to/waveglow_model")
SPEECH_AUDIO = tts_model.synth_speech(TEXT)
```
