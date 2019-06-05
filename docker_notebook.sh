#!/bin/bash

docker run --runtime=nvidia -it --rm -p 8889:8889 \
    -v /home/kamperh/backup/endgame/datasets/buckeye:/data/buckeye \
    -v /home/kamperh/backup/endgame/datasets/zrsc2015/xitsonga_wavs:/data/xitsonga_wavs \
    -v "$(pwd)":/home \
    py3_tf1.13 \
    bash -c "jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8889"
