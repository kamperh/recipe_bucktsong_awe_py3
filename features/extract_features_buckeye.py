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
import features
import utils


def extract_features_for_subset(subset, feat_type, output_fn):
    """
    Extract specified features for a subset.

    The `feat_type` parameter can be "mfcc" or "fbank".
    """

    # Speakers for subset
    speaker_fn = path.join(
        "..", "data", "buckeye_" + subset + "_speakers.list"
        )
    print("Reading:", speaker_fn)
    speakers = set()
    with open(speaker_fn) as f:
        for line in f:
            speakers.add(line.strip())
    print("Speakers:", ", ".join(sorted(speakers)))

    # Raw MFCCs
    feat_dict = {}
    print("Extracting features per speaker:")
    for speaker in sorted(speakers):
        if feat_type == "mfcc":
            speaker_feat_dict = features.extract_mfcc_dir(
                path.join(buckeye_datadir, speaker)
                )
        elif feat_type == "fbank":
            speaker_feat_dict = features.extract_fbank_dir(
                path.join(buckeye_datadir, speaker)
                )
        else:
            assert False, "invalid feature type"
        for wav_key in speaker_feat_dict:
            feat_dict[speaker + "_" + wav_key[3:]] = speaker_feat_dict[wav_key]

    # Read voice activity regions
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    print("Reading:", fa_fn)
    vad_dict = utils.read_vad_from_fa(fa_fn)

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
    
    # Extract MFCCs for the different sets
    mfcc_dir = path.join("mfcc", "buckeye")
    for subset in ["devpart2"]:  #  ["devpart1", "devpart2", "zs"]:  # TO-DO
        if not path.isdir(mfcc_dir):
            os.makedirs(mfcc_dir)
        output_fn = path.join(mfcc_dir, subset + ".dd.npz")
        if not path.isfile(output_fn):
            print(datetime.now())
            print("Extracting MFCCs:", subset)
            extract_features_for_subset(subset, "mfcc", output_fn)
        else:
            print(datetime.now())
            print("Using existing file:", output_fn)
        break  # TO-DO

    # Extract ground truth words of at least 50 frames and 5 characters.
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    list_dir = "lists"
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    list_fn = path.join(list_dir, "buckeye.samediff.list")
    if not path.isfile(list_fn):
        print(datetime.now())
        utils.write_samediff_words(fa_fn, list_fn)
    else:
        print("Using existing file:", list_fn)


    # Extract word segments from the NumPy archives
    for subset in ["devpart2"]:  # TO-DO
        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".samediff.dd.npz")
        if not path.isfile(output_npz_fn):
            print(datetime.now())
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    print(datetime.now())


if __name__ == "__main__":
    main()
