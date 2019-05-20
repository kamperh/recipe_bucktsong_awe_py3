#!/bin/bash

# Install Python dependencies
pip install --user --requirement requirements.txt

# Install speech_dtw
mkdir -p ../src/
git clone https://github.com/kamperh/speech_dtw.git ../src/speech_dtw/
cd ../src/speech_dtw
make
cd -