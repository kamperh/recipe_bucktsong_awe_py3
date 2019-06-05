#!/bin/bash

docker run --runtime=nvidia --rm -it -p 8889:8889 \
    -v /r2d2/backup/endgame/datasets/buckeye:/data/buckeye \
    -v /r2d2/backup/endgame/datasets/zrsc2015/xitsonga_wavs:/data/xitsonga_wavs \
    -v "$(pwd)":/home \
    py3.6_tf1.13 \
    bash -c "ipython notebook --no-browser --ip=0.0.0.0 --allow-root --port=8889"
