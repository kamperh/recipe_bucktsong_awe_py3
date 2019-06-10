#!/usr/bin/env python

"""
Extract MFCC and filterbank features for the NCHLT Xitsonga dataset.

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

from paths import xitsonga_datadir
import features
import utils


def extract_features(feat_type, output_fn):
    """
    Extract specified features.

    The `feat_type` parameter can be "mfcc" or "fbank".
    """

    # Raw features
    feat_dict = {}
    if feat_type == "mfcc":
        feat_dict_wavkey = features.extract_mfcc_dir(xitsonga_datadir)
    elif feat_type == "fbank":
        feat_dict_wavkey = features.extract_fbank_dir(xitsonga_datadir)
    else:
        assert False, "invalid feature type"
    for wav_key in feat_dict_wavkey:
        feat_key = utils.uttlabel_to_uttkey(wav_key)
        feat_dict[feat_key] = feat_dict_wavkey[wav_key]

    # Read voice activity regions
    fa_fn = path.join("..", "data", "xitsonga.wrd")
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

    print(datetime.now())

    # RAW FEATURES

    # Extract MFCCs
    mfcc_dir = path.join("mfcc", "xitsonga")
    if not path.isdir(mfcc_dir):
        os.makedirs(mfcc_dir)
    output_fn = path.join(mfcc_dir, "xitsonga.dd.npz")
    if not path.isfile(output_fn):
        print("Extracting MFCCs")
        extract_features("mfcc", output_fn)
    else:
        print("Using existing file:", output_fn)

    # Extract filterbanks
    fbank_dir = path.join("fbank", "xitsonga")
    if not path.isdir(fbank_dir):
        os.makedirs(fbank_dir)
    output_fn = path.join(fbank_dir, "xitsonga.npz")
    if not path.isfile(output_fn):
        print("Extracting filterbanks")
        extract_features("fbank", output_fn)
    else:
        print("Using existing file:", output_fn)


    # GROUND TRUTH WORD SEGMENTS

    # Create a ground truth word list of at least 50 frames and 5 characters
    fa_fn = path.join("..", "data", "xitsonga.wrd")
    list_dir = "lists"
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    list_fn = path.join(list_dir, "xitsonga.samediff.list")
    if not path.isfile(list_fn):
        utils.write_samediff_words(fa_fn, list_fn)
    else:
        print("Using existing file:", list_fn)

    # Extract word segments from the MFCC NumPy archive
    input_npz_fn = path.join(mfcc_dir, "xitsonga.dd.npz")
    output_npz_fn = path.join(mfcc_dir, "xitsonga.samediff.dd.npz")
    if not path.isfile(output_npz_fn):
        print("Extracting MFCCs for same-different word tokens")
        utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
    else:
        print("Using existing file:", output_npz_fn)

    # Extract word segments from the filterbank NumPy archive
    input_npz_fn = path.join(fbank_dir, "xitsonga.npz")
    output_npz_fn = path.join(fbank_dir, "xitsonga.samediff.npz")
    if not path.isfile(output_npz_fn):
        print("Extracting filterbanks for same-different word tokens")
        utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
    else:
        print("Using existing file:", output_npz_fn)


    # UTD-DISCOVERED WORD SEGMENTS

    # Remove non-VAD regions from the UTD pair list
    input_pairs_fn = path.join("..", "data", "zs_tsonga.fdlps.0.925.pairs.v0")
    output_pairs_fn = path.join("lists", "xitsonga.utd_pairs.list")
    if not path.isfile(output_pairs_fn):
        # Read voice activity regions
        fa_fn = path.join("..", "data", "xitsonga.wrd")
        print("Reading:", fa_fn)
        vad_dict = utils.read_vad_from_fa(fa_fn)

        # Create new pair list
        utils.strip_nonvad_from_pairs(
            vad_dict, input_pairs_fn, output_pairs_fn
            )
    else:
        print("Using existing file:", output_pairs_fn)

    # Create the UTD word list
    list_fn = path.join("lists", "xitsonga.utd_terms.list")
    if not path.isfile(list_fn):
        utils.terms_from_pairs(output_pairs_fn, list_fn)
    else:
        print("Using existing file:", list_fn)

    # Extract UTD segments from the MFCC NumPy archives
    input_npz_fn = path.join(mfcc_dir, "xitsonga.dd.npz")
    output_npz_fn = path.join(mfcc_dir, "xitsonga.utd.dd.npz")
    if not path.isfile(output_npz_fn):
        print("Extracting MFCCs for UTD word tokens")
        utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
    else:
        print("Using existing file:", output_npz_fn)


    # BES-GMM DISCOVERED WORD SEGMENTS

    input_npz_fn = path.join(mfcc_dir, "xitsonga.dd.npz")
    output_npz_fn = path.join(mfcc_dir, "xitsonga.besgmm.dd.npz")
    if not path.isfile(output_npz_fn):
        list_fn = path.join(
            "..", "data", "xitsonga.d18547ee5e.besgmm_terms.txt"
            )
        print("Extracting MFCCs for BES-GMM word tokens")
        utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
    else:
        print("Using existing file:", output_npz_fn)

    print(datetime.now())


if __name__ == "__main__":
    main()
