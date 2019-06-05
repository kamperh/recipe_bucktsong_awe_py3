#!/usr/bin/env python

"""
Train a recurrent variational autoencoder.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
import argparse
import pickle
import hashlib
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(path.join("..", "src"))

from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import batching
import data_io
import samediff
import tflego
import training


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
        "data_dir": path.join("data", "buckeye.mfcc"),
        "train_tag": "utd",                 # "gt", "gt2", "utd", "rnd"
        "max_length": 100,
        "min_length": 50,                   # only used with "rnd" train_tag
        "bidirectional": False,
        "rnn_type": "gru",                  # "lstm", "gru", "rnn"
        "enc_n_hiddens": [400, 400, 400],
        "dec_n_hiddens": [400, 400, 400],
        "n_z": 130,                         # latent dimensionality
        "sigma_sq": 0.00001,                # smaller values: weigh
                                            # reconstruction more
        "learning_rate": 0.001,
        "keep_prob": 1.0,
        "n_epochs": 100,
        "batch_size": 500,
        "n_buckets": 3,
        "extrinsic_usefinal": False,        # if True, during final extrinsic
                                            # evaluation, the final saved model
                                            # will be used (instead of the
                                            # validation best)
        "use_test_for_val": False,
        "n_val_interval": 1,
        "rnd_seed": 1,
    }


#-----------------------------------------------------------------------------#
#                              TRAINING FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def build_vae_from_options_dict(x, x_lengths, options_dict):
    build_latent_func = tflego.build_vae
    latent_func_kwargs = {
        "enc_n_hiddens": [],
        "n_z": options_dict["n_z"],
        "dec_n_hiddens": [options_dict["dec_n_hiddens"][0]],
        "activation": tf.nn.relu
        }
    network_dict = tflego.build_multi_encdec_lazydynamic_latentfunc(
        x, x_lengths, options_dict["enc_n_hiddens"],
        options_dict["dec_n_hiddens"], build_latent_func, latent_func_kwargs,
        rnn_type=options_dict["rnn_type"], keep_prob=options_dict["keep_prob"]
        )
    network_dict["decoder_output"] *= tf.expand_dims(network_dict["mask"], -1)
    # safety
    return network_dict


def train_vae(options_dict):
    """Train and save a VAE."""

    # PRELIMINARY

    print(datetime.now())

    # Output directory
    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    model_dir = path.join(
        "models", path.split(options_dict["data_dir"])[-1] + "." +
        options_dict["train_tag"], options_dict["script"], hash_str
        )
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Model directory:", model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print("Options:", options_dict)

    # Random seeds
    np.random.seed(options_dict["rnd_seed"])
    tf.set_random_seed(options_dict["rnd_seed"])


    # LOAD AND FORMAT DATA

    # Training data
    train_tag = options_dict["train_tag"]
    min_length = None
    if options_dict["train_tag"] == "rnd":
        min_length = options_dict["min_length"]
        train_tag = "all"
    npz_fn = path.join(
        options_dict["data_dir"], "train." + train_tag + ".npz"
        )
    train_x, train_labels, train_lengths, train_keys, train_speakers = (
        data_io.load_data_from_npz(npz_fn, min_length)
        )

    # Validation data
    if options_dict["use_test_for_val"]:
        npz_fn = path.join(options_dict["data_dir"], "test.npz")
    else:
        npz_fn = path.join(options_dict["data_dir"], "val.npz")
    val_x, val_labels, val_lengths, val_keys, val_speakers = (
        data_io.load_data_from_npz(npz_fn)
        )

    # Truncate and limit dimensionality
    max_length = options_dict["max_length"]
    d_frame = 13  # None
    options_dict["n_input"] = d_frame
    print("Limiting dimensionality:", d_frame)
    print("Limiting length:", max_length)
    data_io.trunc_and_limit_dim(train_x, train_lengths, d_frame, max_length)
    data_io.trunc_and_limit_dim(val_x, val_lengths, d_frame, max_length)


    # DEFINE MODEL

    print(datetime.now())
    print("Building model")

    # Model filenames
    intermediate_model_fn = path.join(model_dir, "vae.tmp.ckpt")
    model_fn = path.join(model_dir, "vae.best_val.ckpt")

    # Model graph
    x = tf.placeholder(TF_DTYPE, [None, None, options_dict["n_input"]])
    x_lengths = tf.placeholder(TF_ITYPE, [None])
    network_dict = build_vae_from_options_dict(x, x_lengths, options_dict)
    encoder_states = network_dict["encoder_states"]
    vae = network_dict["latent_layer"]
    z_mean = vae["z_mean"]
    z_log_sigma_sq = vae["z_log_sigma_sq"]
    z = vae["z"]
    y = network_dict["decoder_output"]
    mask = network_dict["mask"]

    # VAE loss
    # reconstruction_loss = tf.reduce_mean(
    #     tf.reduce_sum(tf.reduce_mean(tf.square(x - y), -1), -1) /
    #     tf.reduce_sum(mask, 1)
    #     )  # https://danijar.com/variable-sequence-lengths-in-tensorflow/
    # loss = tflego.vae_loss_gaussian(
    #     x, y, options_dict["sigma_sq"], z_mean, z_log_sigma_sq,
    #     reconstruction_loss=reconstruction_loss
    #     )
    reconstruction_loss = 1./(2*options_dict["sigma_sq"])*tf.reduce_mean(
        tf.reduce_sum(tf.reduce_mean(tf.square(x - y), -1), -1) /
        tf.reduce_sum(mask, 1)
        )  # https://danijar.com/variable-sequence-lengths-in-tensorflow/
    regularisation_loss = -0.5*tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1
        )
    loss = reconstruction_loss + tf.reduce_mean(regularisation_loss)
    # loss = tflego.vae_loss_gaussian(
    #     x, y, options_dict["sigma_sq"], z_mean, z_log_sigma_sq,
    #     reconstruction_loss=reconstruction_loss
    #     )
    optimizer = tf.train.AdamOptimizer(
        learning_rate=options_dict["learning_rate"]
        ).minimize(loss)


    # TRAIN AND VALIDATE

    print(datetime.now())
    print("Training model")

    # Validation function
    def samediff_val(normalise=False):
        # Embed validation
        np.random.seed(options_dict["rnd_seed"])
        val_batch_iterator = batching.SimpleIterator(val_x, len(val_x), False)
        labels = [val_labels[i] for i in val_batch_iterator.indices]
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, val_model_fn)
            for batch_x_padded, batch_x_lengths in val_batch_iterator:
                np_x = batch_x_padded
                np_x_lengths = batch_x_lengths
                np_z = session.run(
                    [z_mean], feed_dict={x: np_x, x_lengths: np_x_lengths}
                    )[0]
                break  # single batch

        embed_dict = {}
        for i, utt_key in enumerate(
                [val_keys[i] for i in val_batch_iterator.indices]):
            embed_dict[utt_key] = np_z[i]

        # Same-different
        if normalise:
            np_z_normalised = (np_z - np_z.mean(axis=0))/np_z.std(axis=0)
            distances = pdist(np_z_normalised, metric="cosine")
            matches = samediff.generate_matches_array(labels)
            ap, prb = samediff.average_precision(
                distances[matches == True], distances[matches == False]
                )
        else:
            distances = pdist(np_z, metric="cosine")
            matches = samediff.generate_matches_array(labels)
            ap, prb = samediff.average_precision(
                distances[matches == True], distances[matches == False]
                )    
        return [prb, -ap]

    # Train VAE
    val_model_fn = intermediate_model_fn
    if options_dict["train_tag"] == "rnd":
        train_batch_iterator = batching.RandomSegmentsIterator(
            train_x, options_dict["batch_size"], options_dict["n_buckets"],
            shuffle_every_epoch=True
            )
    else:
        train_batch_iterator = batching.SimpleBucketIterator(
            train_x, options_dict["batch_size"], options_dict["n_buckets"],
            shuffle_every_epoch=True
            )
    record_dict = training.train_fixed_epochs_external_val(
        options_dict["n_epochs"], optimizer, loss, train_batch_iterator, [x,
        x_lengths], samediff_val, save_model_fn=intermediate_model_fn,
        save_best_val_model_fn=model_fn,
        n_val_interval=options_dict["n_val_interval"]
        )

    # Save record
    record_dict_fn = path.join(model_dir, "record_dict.pkl")
    print("Writing:", record_dict_fn)
    with open(record_dict_fn, "wb") as f:
        pickle.dump(record_dict, f, -1)

    # Save options_dict
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Writing:" + options_dict_fn)
    with open(options_dict_fn, "wb") as f:
        pickle.dump(options_dict, f, -1)


    # FINAL EXTRINSIC EVALUATION

    print ("Performing final validation")
    if options_dict["extrinsic_usefinal"]:
        val_model_fn = intermediate_model_fn
    else:
        val_model_fn = model_fn
    prb, ap = samediff_val(normalise=False)
    ap = -ap
    prb_normalised, ap_normalised = samediff_val(normalise=True)
    ap_normalised = -ap_normalised
    print("Validation AP:", ap)
    print("Validation AP with normalisation:", ap_normalised)
    ap_fn = path.join(model_dir, "val_ap.txt")
    print("Writing:", ap_fn)
    with open(ap_fn, "w") as f:
        f.write(str(ap) + "\n")
        f.write(str(ap_normalised) + "\n")
    print("Validation model:", val_model_fn)

    print(datetime.now())


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0]
        )
    parser.add_argument(
        "--data_dir", type=str,
        help="load data from this directory (default: %(default)s)",
        default=default_options_dict["data_dir"]
        )
    parser.add_argument(
        "--n_epochs", type=int,
        help="number of epochs of training (default: %(default)s)",
        default=default_options_dict["n_epochs"]
        )
    parser.add_argument(
        "--batch_size", type=int,
        help="size of mini-batch (default: %(default)s)",
        default=default_options_dict["batch_size"]
        )
    parser.add_argument(
        "--train_tag", type=str, choices=["gt", "gt2", "utd", "rnd"],
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_tag"]
        )
    parser.add_argument(
        "--extrinsic_usefinal", action="store_true",
        help="if set, during final extrinsic evaluation, the final saved "
        "model will be used instead of the validation best (default: "
        "%(default)s)",
        default=default_options_dict["extrinsic_usefinal"]
        )
    parser.add_argument(
        "--use_test_for_val", action="store_true",
        help="if set, use the test data for validation (cheating, so only "
        "use when doing final evaluation, otherwise this is cheating) "
        "(default: %(default)s)",
        default=default_options_dict["use_test_for_val"]
        )
    parser.add_argument(
        "--rnd_seed", type=int, help="random seed (default: %(default)s)",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--sigma_sq", type=float,
        help="the fixed variance of the output Gaussian "
        "(default: %(default)s)", default=default_options_dict["sigma_sq"]
        )

    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(args.data_dir)
    
    # Set options
    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_vae"
    options_dict["data_dir"] = args.data_dir
    options_dict["n_epochs"] = args.n_epochs
    options_dict["batch_size"] = args.batch_size
    options_dict["sigma_sq"] = args.sigma_sq
    options_dict["extrinsic_usefinal"] = args.extrinsic_usefinal
    options_dict["use_test_for_val"] = args.use_test_for_val
    options_dict["train_tag"] = args.train_tag
    options_dict["rnd_seed"] = args.rnd_seed

    # Train model
    train_vae(options_dict)


if __name__ == "__main__":
    main()
