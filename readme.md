Unsupervised Acoustic Word Embeddings on Buckeye English and NCHLT Xitsonga
===========================================================================

Overview
--------
Unsupervised acoustic word embedding (AWE) approaches are implemented and
evaluated on the Buckeye English and NCHLT Xitsonga speech datasets. The
experiments are described in:

- H. Kamper, "Truly unsupervised acoustic word embeddings using weak top-down
  constraints in encoder-decoder models," in *Proc. ICASSP*, 2019.
  [[arXiv](https://arxiv.org/abs/1811.00403)]

Please cite this paper if you use the code.

**TODO: Mention if this recipe replaces an older one.**


Disclaimer
----------
The code provided here is not pretty. But I believe that research should be
reproducible. I provide no guarantees with the code, but please let me know if
you have any problems, find bugs or have general comments.


Download datasets
-----------------
Portions of the Buckeye English and NCHLT Xitsonga corpora are used. The whole
Buckeye corpus is used and a portion of the NCHLT data. These can be downloaded
from:

- Buckeye corpus:
  [buckeyecorpus.osu.edu](http://buckeyecorpus.osu.edu/)
- NCHLT Xitsonga portion:
  [www.zerospeech.com](http://www.lscp.net/persons/dupoux/bootphon/zerospeech2014/website/page_4.html).
  This requires registration for the challenge.

From the complete Buckeye corpus we split off several subsets: the sets
labelled as `devpart1` and `zs` respectively correspond to the `English1` and
`English2` sets in [Kamper et al., 2016](http://arxiv.org/abs/1606.06950). We
use the Xitsonga dataset provided as part of the Zero Speech Challenge 2015 (a
subset of the NCHLT data).


Create and run Docker image
---------------------------
This recipe provides a Docker image containing all the required dependencies.
The recipe can be run without Docker, but then the dependencies need to be
installed separately (see below). To use the Docker image, you need to:

- Install [Docker](https://docs.docker.com/install/) and follow the [post
  installation
  steps](https://docs.docker.com/install/linux/linux-postinstall/).
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

To build the Docker image, run:

    cd docker
    docker build -f Dockerfile.gpu -t py3.6_tf1.13 .
    cd ..

The remaining steps in this recipe can be run in a container in interactive
mode. The dataset directories will also need to be mounted. To run a container
in interactive mode with the mounted directories, run:

    docker run --runtime=nvidia -it --rm -u $(id -u):$(id -g) -p 8887:8887 \
        -v /r2d2/backup/endgame/datasets/buckeye:/data/buckeye \
        -v /r2d2/backup/endgame/datasets/zrsc2015/xitsonga_wavs:/data/xitsonga_wavs \
        -v "$(pwd)":/home \
        py3.6_tf1.13

Alternatively, run `./docker.sh`, which executes the above command and starts
an interactive container. To directly start a Jupyter notebook in a container,
run `./docker_notebook.sh`.


If not using Docker: Install dependencies
-----------------------------------------
If you are not using Docker, install the following dependencies:

- [Python 3](https://www.python.org/downloads/)
- [TensorFlow 1.13.1](https://www.tensorflow.org/)
- [LibROSA](http://librosa.github.io/librosa/)
- [Cython](https://cython.org/)
- [tqdm](https://tqdm.github.io/)


Extract speech features
-----------------------
Update the paths in `paths.py` to point to the datasets. If you are using
docker, `paths.py` will already point to the mounted directories. Extract MFCC
and filterbank features in the `features/` directory as follows:

    cd features
    ./extract_features_buckeye.py
    ./extract_features_xitsonga.py


Evaluate frame-level features using the same-different task
-----------------------------------------------------------
This is optional. To perform frame-level same-different evaluation based on
dynamic time warping (DTW), follow [samediff/readme.md](samediff/readme.md).


Obtain downsampled acoustic word embeddings
-------------------------------------------
Extract and evaluate downsampled acoustic word embeddings by running the steps
in [downsample/readme.md](downsample/readme.md).


Train neural acoustic word embeddings
-------------------------------------


Notebooks
---------


Testing
-------
In the root project directory, run `make test` to run unit tests.


License
-------
The code is distributed under the Creative Commons Attribution-ShareAlike
license ([CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)).
