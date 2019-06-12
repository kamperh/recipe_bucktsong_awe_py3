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

    # Raw features
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

    print(datetime.now())

    # RAW FEATURES

    # Extract MFCCs for the different sets
    mfcc_dir = path.join("mfcc", "buckeye")
    for subset in ["devpart1", "devpart2", "zs"]:
        if not path.isdir(mfcc_dir):
            os.makedirs(mfcc_dir)
        output_fn = path.join(mfcc_dir, subset + ".dd.npz")
        if not path.isfile(output_fn):
            print("Extracting MFCCs:", subset)
            extract_features_for_subset(subset, "mfcc", output_fn)
        else:
            print("Using existing file:", output_fn)

    # Extract filterbanks for the different sets
    fbank_dir = path.join("fbank", "buckeye")
    for subset in ["devpart1", "devpart2", "zs"]:
        if not path.isdir(fbank_dir):
            os.makedirs(fbank_dir)
        output_fn = path.join(fbank_dir, subset + ".npz")
        if not path.isfile(output_fn):
            print("Extracting filterbanks:", subset)
            extract_features_for_subset(subset, "fbank", output_fn)
        else:
            print("Using existing file:", output_fn)


    # GROUND TRUTH WORD SEGMENTS

    # Create a ground truth word list of at least 50 frames and 5 characters
    fa_fn = path.join("..", "data", "buckeye_english.wrd")
    list_dir = "lists"
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    list_fn = path.join(list_dir, "buckeye.samediff.list")
    if not path.isfile(list_fn):
        utils.write_samediff_words(fa_fn, list_fn)
    else:
        print("Using existing file:", list_fn)

    # Extract word segments from the MFCC NumPy archives
    for subset in ["devpart1", "devpart2", "zs"]:
        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".samediff.dd.npz")
        if not path.isfile(output_npz_fn):
            print("Extracting MFCCs for same-different word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    # Extract word segments from the filterbank NumPy archives
    for subset in ["devpart1", "devpart2", "zs"]:
        input_npz_fn = path.join(fbank_dir, subset + ".npz")
        output_npz_fn = path.join(fbank_dir, subset + ".samediff.npz")
        if not path.isfile(output_npz_fn):
            print(
                "Extracting filterbanks for same-different word tokens:",
                subset
                )
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    # Create a ground truth word list of at least 39 frames and 4 characters
    list_fn = path.join(list_dir, "buckeye.samediff2.list")
    if not path.isfile(list_fn):
        utils.write_samediff_words(fa_fn, list_fn, min_frames=39, min_chars=4)
    else:
        print("Using existing file:", list_fn)

    # Extract word segments from the MFCC NumPy archives
    for subset in ["devpart1"]:  # , "devpart2", "zs"]:
        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".samediff2.dd.npz")
        if not path.isfile(output_npz_fn):
            print("Extracting MFCCs for same-different word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    # UTD-DISCOVERED WORD SEGMENTS

    # Remove non-VAD regions from the UTD pair list
    input_pairs_fn = path.join("..", "data", "buckeye.fdlps.0.93.pairs")
    output_pairs_fn = path.join("lists", "buckeye.utd_pairs.list")
    if not path.isfile(output_pairs_fn):
        # Read voice activity regions
        fa_fn = path.join("..", "data", "buckeye_english.wrd")
        print("Reading:", fa_fn)
        vad_dict = utils.read_vad_from_fa(fa_fn)

        # Create new pair list
        utils.strip_nonvad_from_pairs(
            vad_dict, input_pairs_fn, output_pairs_fn
            )
    else:
        print("Using existing file:", output_pairs_fn)

    # Create the UTD word list
    list_fn = path.join("lists", "buckeye.utd_terms.list")
    if not path.isfile(list_fn):
        utils.terms_from_pairs(output_pairs_fn, list_fn)
    else:
        print("Using existing file:", list_fn)

    # Extract UTD segments from the MFCC NumPy archives
    for subset in ["devpart1"]:

        # Extract pair and term list for speakers in subset
        speaker_fn = path.join(
            "..", "data", "buckeye_{}_speakers.list".format(subset)
            )
        input_pairs_fn = output_pairs_fn
        output_pairs_fn = path.join("lists", "devpart1.utd_pairs.list")
        if not path.isfile(output_pairs_fn):
            utils.pairs_for_speakers(
                speaker_fn, input_pairs_fn, output_pairs_fn
                )
        else:
            print("Using existing file:", output_pairs_fn)
        list_fn = path.join("lists", "devpart1.utd_terms.list")
        if not path.isfile(list_fn):
            utils.terms_from_pairs(output_pairs_fn, list_fn)
        else:
            print("Using existing file:", list_fn)

        # Extract UTD segments
        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".utd.dd.npz")
        if not path.isfile(output_npz_fn):
            print("Extracting MFCCs for UTD word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)


    # BES-GMM DISCOVERED WORD SEGMENTS

    for subset in ["devpart1"]:
        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".besgmm.dd.npz")
        if not path.isfile(output_npz_fn):
            list_fn = path.join(
                "..", "data", "buckeye_devpart1.52e70ca864.besgmm_terms.txt"
                )
            print("Extracting MFCCs for BES-GMM word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(mfcc_dir, subset + ".besgmm1.dd.npz")
        if not path.isfile(output_npz_fn):
            list_fn = path.join(
                "..", "data",
                "buckeye_devpart1.52e70ca864.besgmm_terms_filt1.txt"
                )
            print("Extracting MFCCs for BES-GMM word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

        for tag in ["2", "3", "4"]:
            pairs_fn = path.join(
                "..", "data",
                "buckeye_devpart1.52e70ca864.besgmm_pairs_filt" + tag + ".txt"
                )
            list_fn = path.join(
                "lists", "buckeye_devpart1.52e70ca864.besgmm_terms_filt" + tag
                + ".txt"
                )
            if not path.isfile(list_fn):
                utils.terms_from_pairs(pairs_fn, list_fn)
            else:
                print("Using existing file:", list_fn)
            input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
            output_npz_fn = path.join(mfcc_dir, subset + ".besgmm" + tag +
            ".dd.npz")
            if not path.isfile(output_npz_fn):
                print("Extracting MFCCs for BES-GMM word tokens:", subset)
                utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
            else:
                print("Using existing file:", output_npz_fn)

        input_npz_fn = path.join(mfcc_dir, subset + ".dd.npz")
        output_npz_fn = path.join(
            mfcc_dir, subset + ".besgmm_mindur0.425.dd.npz"
            )
        if not path.isfile(output_npz_fn):
            list_fn = path.join(
                "..", "data",
                "buckeye_devpart1.52e70ca864.besgmm_terms.mindur0.425.txt"
                )
            print("Extracting MFCCs for BES-GMM word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)

    print(datetime.now())


if __name__ == "__main__":
    main()
