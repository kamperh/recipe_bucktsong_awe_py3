#!/usr/bin/env python

"""
Run a script with different options given comma-separated arguments.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016, 2018, 2019
"""

from __future__ import division
from __future__ import print_function
import argparse
import itertools
import subprocess
import sys

sweep_options = [
    "rnd_seed", "n_hiddens", "enc_n_layers", "dec_n_layers", "ae_n_epochs",
    "cae_n_epochs", "train_tag", "ae_batch_size", "cae_batch_size",
    "batch_size", "sigma_sq", "n_epochs"
    ]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "script", type=str, help="script to run",
        choices=["train_cae", "train_vae"]
        )
    parser.add_argument(
        "--static_args", type=str,
        help="arguments that does not need to be swept (normally given in "
        "quotes)"
        )
    for option in sweep_options:
        parser.add_argument("--" + option, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    sweep_options_list = []
    sweep_options_value_list = []
    for option in sweep_options:
        attr = getattr(args, option)
        if attr is not None:
            sweep_options_list.append(option)
            sweep_options_value_list.append(attr.split(","))

    print("-"*79)
    for cur_sweep_params_values in itertools.product(
            *sweep_options_value_list):
        if args.static_args is None:
            option_str = ""
        else:
            option_str = args.static_args + " "
        for i in range(len(sweep_options_list)):
            cur_option = sweep_options_list[i]
            cur_option_value = cur_sweep_params_values[i]
            option_str += "--" + cur_option + " "
            option_str += cur_option_value + " "
        cmd = "./" + args.script + ".py " + option_str
        print(cmd)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    print("-"*79)


if __name__ == "__main__":
    main()
