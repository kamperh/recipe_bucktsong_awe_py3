#!/usr/bin/env python

"""
Perform test-set evaluation on models in output produced by sweep.py.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from os import path
import argparse
import pickle
import numpy as np
import subprocess
import sys


shell = lambda command: subprocess.check_output(
    command, stderr=subprocess.STDOUT, shell=True
    )


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

    model_tag = ""
    if "train_cae" in model_dir:
        if (not path.isfile(path.join(model_dir, "cae.best_val.ckpt"))
                and not path.isfile(path.join(model_dir,
                "cae.best_val.ckpt.index"))):
            model_tag = "ae"
        else:
            model_tag = "cae"
    elif "train_vae" in model_dir:
        model_tag = "vae"

    # Apply models
    for model_dir in model_dirs:
        command = (
            "./apply_model.py " + path.join(model_dir, model_tag +
            ".best_val.ckpt") +
            " test"
            )
        print("Running:", command)
        shell(command)

    # Evaluate AP
    for model_dir in model_dirs:
        command = (
            "./eval_samediff.py " + path.join(model_dir,
            model_tag + ".best_val.test.npz") + 
            " | grep \"Average\" | awk '{print $3}' > " +
            path.join(model_dir, "test_ap.txt")
            )
        print("Running:", command)
        shell(command)

    # Evaluate AP with normalisation
    for model_dir in model_dirs:
        command = (
            "./eval_samediff.py --mvn " + path.join(model_dir,
            model_tag + ".best_val.test.npz") + 
            " | grep \"Average\" | awk '{print $3}' >> " +
            path.join(model_dir, "test_ap.txt")
            )
        print("Running:", command)
        shell(command)


if __name__ == "__main__":
    main()
