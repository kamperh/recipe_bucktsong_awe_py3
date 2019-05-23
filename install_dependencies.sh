#!/bin/bash

# Install Python dependencies
pip install --user --requirement requirements.txt

# Install speech_dtw
if [ ! -d ../src ]; then
    mkdir ../src/
fi
if [ ! -d ../src/speech_dtw/ ]; then
    git clone https://github.com/kamperh/speech_dtw.git ../src/speech_dtw/
    cd ../src/speech_dtw
    make
    cd -
fi
