Downsampled Acoustic Word Embeddings
====================================


Overview
--------
MFCCs are downsampled to obtain acoustic word embeddings. These are evaluated
using same-different evaluation.


Downsampling
------------
Perform downsampling on MFCCs without deltas:

    # Devpart2
    n_samples=10
    mkdir -p exp/devpart2
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/buckeye/devpart2.samediff.dd.npz \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}

    # ZeroSpeech
    n_samples=10
    mkdir -p exp/zs
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/buckeye/zs.samediff.dd.npz \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}

    # Xitsonga
    n_samples=10
    mkdir -p exp/xitsonga
    ./downsample.py --technique resample --frame_dims 13 \
        ../features/mfcc/xitsonga/xitsonga.samediff.dd.npz \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz \
        ${n_samples}


Evaluation
----------
Evaluate and analyse downsampled MFCCs without deltas:

    # Devpart2
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        because,yknow,people,something,anything,education,situation \
        exp/devpart2/samediff.mfcc.downsample_${n_samples}.npz

    # ZeroSpeech
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        because,yknow,people,something,anything,education,situation \
        exp/zs/samediff.mfcc.downsample_${n_samples}.npz

    # Xitsonga
    n_samples=10
    ./eval_samediff.py --mvn \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz
    ./analyse_embeds.py --normalize --word_type \
        kombisa,swilaveko,kahle,swinene,xiyimo,fanele,naswona,xikombelo \
        exp/xitsonga/samediff.mfcc.downsample_${n_samples}.npz


Results
-------
Devpart2 downsampled MFCCs without deltas (dimensionality=130):

    Average precision: 0.22061737340264037
    Precision-recall breakeven: 0.2679401135776975

ZeroSpeech downsampled MFCCs without deltas + mvn (dimensionality=130):

    Average precision: 0.19394796913284967
    Precision-recall breakeven: 0.2538620273479665

Xitsonga downsampled MFCCs without deltas (dimensionality=130):

    Average precision: 0.12970181712047946
    Precision-recall breakeven: 0.20233273934877694

