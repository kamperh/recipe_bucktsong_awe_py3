#!/bin/bash

docker run --runtime=nvidia -it --rm -u $(id -u):$(id -g) -p 8887:8887 \
    -v /home/kamperh/backup/endgame/datasets/buckeye:/data/buckeye \
    -v /home/kamperh/backup/endgame/datasets/zrsc2015/xitsonga_wavs:/data/xitsonga_wavs \
    -v "$(pwd)":/home \
    py3.6_tf1.13
