#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "klepto==0.1.6",
    "numpy~=1.16.4",
    "inflect==0.2.5",
    "librosa==0.6.0",
    "scipy~=1.3.0",
    "Unidecode==1.0.22",
    "torch~=1.1.0",
]

extra_requirements = {"playback": ["PyAudio==0.2.11"]}

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

packages = find_packages()

setup(
    author="Malar Kannan",
    author_email="malarkannan.invention@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Taco2 TTS package.",
    install_requires=requirements,
    extras_require=extra_requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="tacotron2 tts",
    name="taco2-tts",
    packages=packages,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/malarinv/tacotron2",
    version="0.3.0",
    zip_safe=False,
    entry_points={"console_scripts": ("tts_debug = taco2.tts:main",)},
)
