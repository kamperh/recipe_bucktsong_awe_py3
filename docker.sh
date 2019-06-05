#!/bin/bash

docker run --runtime=nvidia \
    -v /r2d2/backup/endgame/datasets/buckeye:/data/buckeye \
    -v /r2d2/backup/endgame/datasets/zrsc2015/xitsonga_wavs:/data/xitsonga_wavs \
    -v "$(pwd)":/home -it --rm -p 8887:8887 py3.6_tf1.13
