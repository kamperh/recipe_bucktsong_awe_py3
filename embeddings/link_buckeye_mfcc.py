#!/usr/bin/env python

"""
Create links to the MFCC files.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from os import path
import numpy as np
import os

relative_features_dir = path.join("..", "..", "..", "features")
output_dir = path.join("data", "buckeye.mfcc")


def main():

    # Create output directory
    if not path.isdir(output_dir):
        os.makedirs(output_dir)

    # Training: All complete utterances
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart1.dd.npz"
        )
    link_fn = path.join(output_dir, "train.all.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Training: Ground truth words
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart1.samediff.dd.npz"
        )
    link_fn = path.join(output_dir, "train.gt.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Training: Ground truth words (larger set)
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart1.samediff2.dd.npz"
        )
    link_fn = path.join(output_dir, "train.gt2.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Training: UTD discovered words
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart1.utd.dd.npz"
        )
    link_fn = path.join(output_dir, "train.utd.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Training: BES-GMM discovered words
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart1.besgmm.dd.npz"
        )
    link_fn = path.join(output_dir, "train.besgmm.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Training: BES-GMM discovered words
    for tag in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "sd1", "sd2", "sd3", "sd4"]:
        npz_fn = path.join(
            relative_features_dir, "mfcc", "buckeye", "devpart1.besgmm" + tag +
            ".dd.npz"
            )
        link_fn = path.join(output_dir, "train.besgmm" + tag + ".npz")
        assert (
            path.isfile(path.join(output_dir, npz_fn))
            ), "missing file: {}".format(path.join(output_dir, npz_fn))
        if not path.isfile(link_fn):
            print("Linking:", npz_fn, "to", link_fn)
            os.symlink(npz_fn, link_fn)

    # Training: BES-GMM discovered words
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye",
        "devpart1.besgmm_mindur0.425.dd.npz"
        )
    link_fn = path.join(output_dir, "train.besgmm_mindur0.425.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Validation
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "devpart2.samediff.dd.npz"
        )
    link_fn = path.join(output_dir, "val.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)

    # Test
    npz_fn = path.join(
        relative_features_dir, "mfcc", "buckeye", "zs.samediff.dd.npz"
        )
    link_fn = path.join(output_dir, "test.npz")
    assert (
        path.isfile(path.join(output_dir, npz_fn))
        ), "missing file: {}".format(path.join(output_dir, npz_fn))
    if not path.isfile(link_fn):
        print("Linking:", npz_fn, "to", link_fn)
        os.symlink(npz_fn, link_fn)


if __name__ == "__main__":
    main()
