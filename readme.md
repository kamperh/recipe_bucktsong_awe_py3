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
`English2` sets in [Kamper et al., 2016](http://arxiv.org/abs/1606.06950). More
details of which speakers are found in which set is also given at the end of
[features/readme.md](features/readme.md). We use the Xitsonga dataset provided
as part of the Zero Speech Challenge 2015 (a subset of the NCHLT data).


Create Docker container
-----------------------
**TODO: Add py3_tf1.13 Docker image. Rename section.**


If not using Docker: Install dependencies
-----------------------------------------
If you are not using the Docker image, you will need to install the following
dependencies:

- [Python 3](TO-DO)
- librosa
- etc. TO-DO

You can use the following steps to create a virtual environment and install
these dependencies:

    python3 -m venv ~/tools/py3_tf1.13
    source ~/tools/py3_tf1.13/bin/activate
    ./install_dependencies.sh  # TO-DO: Update requirements.txt



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


Train neural acoustic word embeddings
-------------------------------------


Notebooks
---------


Testing
-------
In the root project directory, run `make test` to run unit tests.

