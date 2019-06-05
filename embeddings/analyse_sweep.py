#!/usr/bin/env python

"""
Analyse output produced by sweep.py.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from os import path
import argparse
import pickle
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "log_fn", type=str, help="the file with the output from sweep.py"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Model directories
    model_dirs = []
    with open(args.log_fn) as f:
        for line in f:
            if "Model directory: " in line:
                model_dir = line.strip().replace("Model directory: ", "")
                model_dirs.append(model_dir)

    # Get AP and options dictionary from each directory
    model_records = []
    option_dicts = []
    for model_dir in model_dirs:

        record = {}
        print("Reading:", model_dir)

        # Options dictionary
        fn = path.join(model_dir, "options_dict.pkl")
        if not path.isfile(fn):
            print("Skipping: missing files")
            continue
        with open(fn, "rb") as f:
            options_dict = pickle.load(f)
        record["rnd_seed"] = options_dict["rnd_seed"]
        del options_dict["rnd_seed"]
        record["options_dict"] = options_dict

        # Evaluations
        fn = path.join(model_dir, "val_ap.txt")
        if not path.isfile(fn):
            print("Skipping: missing files")
            continue
        with open(fn, "r") as f:
            ap = float(f.readline().strip())
            ap_normalised = float(f.readline().strip())
            record["val_ap"] = ap
            record["val_ap_normalised"] = ap_normalised
        fn = path.join(model_dir, "test_ap.txt")
        if path.isfile(fn):
            with open(fn, "r") as f:
                ap = float(f.readline().strip())
                ap_normalised = float(f.readline().strip())
                record["test_ap"] = ap
                record["test_ap_normalised"] = ap_normalised

        # Add record
        if options_dict not in option_dicts:
            option_dicts.append(options_dict)
        model_records.append(record)

    # Print analysis ordered by options ignoring seeds
    for options_dict in sorted(option_dicts):
        print()
        print("-"*79)
        val_ap_list = []
        val_ap_normalised_list = []
        test_ap_list = []
        test_ap_normalised_list = []
        print(options_dict)
        for record in model_records:
            if record["options_dict"] == options_dict:
                print(
                    "Validation AP (rnd_seed={:d}): {:.4f}".format(
                    record["rnd_seed"], record["val_ap"])
                    )
                print(
                    "Validation AP with normalisation: {:.4f}".format(
                    record["val_ap_normalised"])
                    )
                val_ap_list.append(record["val_ap"])
                val_ap_normalised_list.append(record["val_ap_normalised"])
                if "test_ap" in record:
                    test_ap_list.append(record["test_ap"])
                if "test_ap_normalised" in record:
                    test_ap_normalised_list.append(
                        record["test_ap_normalised"]
                        )
        if len(val_ap_list) > 1 or len(test_ap_list) > 1:
            print("-"*79)
        if len(val_ap_list) > 1:
            print(
                "Validation AP mean: {:.4f} (+- {:.4f})".format(
                np.mean(val_ap_list), np.std(val_ap_list))
                )
            print(
                "Validation AP with normalisation mean: "
                "{:.4f} (+- {:.4f})".format(np.mean(val_ap_normalised_list),
                np.std(val_ap_normalised_list))
                )
        if len(test_ap_list) > 1:
            print(
                "Test AP mean: {:.4f} (+- {:.4f})".format(
                np.mean(test_ap_list), np.std(test_ap_list))
                )
            print(
                "Test AP with normalisation mean: "
                "{:.4f} (+- {:.4f})".format(np.mean(test_ap_normalised_list),
                np.std(test_ap_normalised_list))
                )
        print("-"*79)

    """
    # Get validation AP in each directory
    print("-"*79)
    ap_list = []
    ap_list_normalised = []
    for model_dir in model_dirs:
        print("Model:", model_dir)
        fn = path.join(model_dir, "val_ap.txt")
        with open(fn, "r") as f:
            ap = float(f.readline().strip())
            ap_list.append(ap)
            print("Validation AP: {:.4f}".format(ap))
            ap_normalised = float(f.readline().strip())
            ap_list_normalised.append(ap_normalised)
            print("Validation AP with normalisation: {:.4f}".format(ap_normalised))
        fn = path.join(model_dir, "options_dict.pkl")
        with open(fn, "rb") as f:
            options_dict = pickle.load(f)
            print(options_dict)
        print("-"*79)

    print()
    print(
        "Validation AP mean: {:.4f} (+- {:.4f})".format(np.mean(ap_list),
        np.std(ap_list))
        )
    print(
        "Validation AP with normalisation mean: {:.4f} (+- {:.4f})".format(
        np.mean(ap_list_normalised), np.std(ap_list_normalised))
        )
    """


if __name__ == "__main__":
    main()
