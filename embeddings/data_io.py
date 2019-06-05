"""
Data input and output functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018
"""

from __future__ import division
from __future__ import print_function
import numpy as np


def load_data_from_npz(npz_fn, min_length=None):
    print("Reading:", npz_fn)
    npz = np.load(npz_fn)
    x = []
    labels = []
    speakers = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in sorted(npz):
        if min_length is not None and len(npz[utt_key]) <= min_length:
            continue
        keys.append(utt_key)
        x.append(npz[utt_key])
        word = utt_key.split("_")[0]
        speaker = utt_key.split("_")[1][:3]
        labels.append(word)
        speakers.append(speaker)
        lengths.append(npz[utt_key].shape[0])
        n_items += 1
    print("No. items:", n_items)
    print("E.g. item shape:", x[0].shape)
    return (x, labels, lengths, keys, speakers)


def trunc_and_limit_dim(x, lengths, d_frame, max_length):
    for i, seq in enumerate(x):
        x[i] = x[i][:max_length, :d_frame]
        lengths[i] = min(lengths[i], max_length)
