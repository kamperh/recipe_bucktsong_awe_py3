#!/usr/bin/env python

"""
Extract MFCC and filterbank features for the Buckeye dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys

sys.path.append("..")

from paths import buckeye_datadir
from utils import read_vad_from_fa, write_samediff_words
import features



def extract_features_for_subset(subset, feat_type, output_fn):
    """
    Extract specified features for a subset.

    The `feat_type` parameter can be "mfcc" or "fbank".
    """
    assert feat_type in ["mfcc", "fbank"]

    # Speakers for subset
    speaker_fn = path.join(
        "..", "data", "buckeye_" + subset + "_speakers.list"
        )
    print("Reading:", speaker_fn)
    speakers = set()
    with open(speaker_fn) as f:
        for line in f:
            speakers.add(line.strip())
    print("Speakers:", sorted(speakers))

    # Raw MFCCs
    feat_dict = {}
    print("Extracting features per speaker:")
    for speaker in sorted(speakers):
        speaker_feat_dict = features.extract_mfcc_dir(
            path.join(buckeye_datadir, speaker)
            )
        for wav_key in speaker_feat_dict:
            feat_dict[speaker + "_" + wav_key[3:]] = speaker_feat_dict[wav_key]
        break  # TO-DO

    # Read voice activity regions
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    print("Reading:", fa_fn)
    vad_dict = read_vad_from_fa(fa_fn)

    # Only keep voice active regions
    print("Extracting VAD regions:")
    feat_dict = features.extract_vad(feat_dict, vad_dict)

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)


def main():
    
    print(datetime.now())

    # Extract MFCCs for the different sets
    output_dir = path.join("mfcc", "buckeye")
    for subset in ["devpart2"]:  #  ["devpart1", "devpart2", "zs"]:
        if not path.isdir(output_dir):
            os.makedirs(output_dir)
        output_fn = path.join(output_dir, subset + ".dd.npz")
        if not path.isfile(output_fn):
            print("Extracting MFCCs:", subset)
            extract_features_for_subset(subset, "mfcc", output_fn)
        break

    # Extract ground truth words of at least 50 frames and 5 characters.
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    output_dir = "lists"
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    output_fn = path.join(output_dir, "buckeye.samediff.list")
    if not path.isfile(output_fn):
        write_samediff_words(fa_fn, output_fn)



    print(datetime.now())


if __name__ == "__main__":
    main()
