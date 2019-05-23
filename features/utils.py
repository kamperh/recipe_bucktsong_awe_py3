"""
Utility functions

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

from tqdm import tqdm
import numpy as np


def uttlabel_to_uttkey(utterance):
    if utterance.startswith("nchlt"):
        # Xitsonga
        utt_split = utterance.split("_")
        speaker = utt_split.pop(2)
        utt_key = speaker + "_" + "-".join(utt_split)
    else:
        # Buckeye
        utt_key = utterance[0:3] + "_" + utterance[3:]
    return utt_key


def read_vad_from_fa(fa_fn, frame_indices=True):
    """
    Read voice activity detected (VAD) regions from a forced alignment file.

    The dictionary has utterance labels as keys and as values the speech
    regions as lists of tuples of (start, end) frame, with the end excluded.
    """
    vad_dict = {}
    prev_utterance = ""
    prev_token_label = ""
    prev_end_time = -1
    start_time = -1
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start_token, end_token, token_label = line.strip(
                ).split()
            start_token = float(start_token)
            end_token = float(end_token)
            # utterance = utterance.replace("_", "-")
            utt_key = uttlabel_to_uttkey(utterance)
            # utt_key = utterance[0:3] + "_" + utterance[3:]
            if utt_key not in vad_dict:
                vad_dict[utt_key] = []

            if token_label in ["SIL", "SPN"]:
                continue
            if prev_end_time != start_token or prev_utterance != utterance:
                if prev_end_time != -1:
                    utt_key = uttlabel_to_uttkey(prev_utterance)
                    # utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
                    if frame_indices:
                        # Convert time to frames
                        start = int(round(start_time * 100))
                        end = int(round(prev_end_time * 100)) + 1
                        vad_dict[utt_key].append((start, end))
                    else:
                        vad_dict[utt_key].append(
                            (start_time, prev_end_time)
                            )
                start_time = start_token

            prev_end_time = end_token
            prev_token_label = token_label
            prev_utterance = utterance

        utt_key = uttlabel_to_uttkey(prev_utterance)
        # utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
        if frame_indices:
            # Convert time to frames
            start = int(round(start_time * 100))
            end = int(round(prev_end_time * 100)) + 1  # end index excluded
            vad_dict[utt_key].append((start, end))
        else:
            vad_dict[utt_key].append((start_time, prev_end_time))        
    return vad_dict


def write_samediff_words(fa_fn, output_fn):
    """
    Extract ground truth types of at least 50 frames and 5 characters.

    Words are extracted from the forced alignment file `fa_fn` and written to
    the word list file `output_fn`.
    """
    print("Reading:", fa_fn)
    words = []
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start, end, label = line.strip().split()
            # utterance = utterance.replace("_", "-")
            start = float(start)
            end = float(end)
            if label in ["SIL", "SPN"]:
                continue
            words.append((utterance, label, (start, end)))

    print("Finding same-different word tokens")
    words_50fr5ch = []
    for utterance, label, (start, end) in words:
        start_frame = int(round(float(start) * 100))
        end_frame = int(round(float(end) * 100))
        if end_frame - start_frame >= 50 and len(label) >= 5:
            words_50fr5ch.append((utterance, label, (start_frame, end_frame)))
    print("No. tokens:", len(words_50fr5ch), "out of", len(words))

    # if not path.isdir(output_dir):
    #     os.makedirs(output_dir)
    print("Writing:", output_fn)
    with open(output_fn, "w") as f:
        for utterance, label, (start, end) in words_50fr5ch:
            utt_key = uttlabel_to_uttkey(utterance)
            # utt_key = utterance[0:3] + "_" + utterance[3:]
            f.write(
                label + "_" + utt_key + "_%06d-%06d\n" % (int(round(start)),
                int(round(end)) + 1)
                )


def segments_from_npz(input_npz_fn, segments_fn, output_npz_fn):
    """
    Cut segments from a NumPy archive and save in a new archive.

    As keys, the archives use the format "label_spkr_utterance_start-end".
    """

    # Read the .npz file
    print("Reading npz:", input_npz_fn)
    input_npz = np.load(input_npz_fn)

    # Create input npz segments dict
    utterance_segs = {}  # utterance_segs["s08_02b_029657-029952"]
                         # is (29657, 29952)
    for key in input_npz.keys():
        utterance_segs[key] = tuple(
            [int(i) for i in key.split("_")[-1].split("-")]
            )

    # Create target segments dict
    print("Reading segments:", segments_fn)
    target_segs = {}  # target_segs["years_s01_01a_004951-005017"]
                      # is ("s01_01a", 4951, 5017)
    for line in open(segments_fn):
        line_split = line.split("_")
        utterance = line_split[-3] + "_" + line_split[-2]
        start, end = line_split[-1].split("-")
        start = int(start)
        end = int(end)
        target_segs[line.strip()] = (utterance, start, end)

    print("Extracting segments:")
    output_npz = {}
    n_target_segs = 0
    for target_seg_key in tqdm(sorted(target_segs)):
        utterance, target_start, target_end = target_segs[target_seg_key]
        for utterance_key in [
                i for i in utterance_segs.keys() if i.startswith(utterance)]:
            utterannce_start, utterance_end = utterance_segs[utterance_key]
            if (target_start >= utterannce_start and target_start <=
                    utterance_end):
                start = target_start - utterannce_start
                end = target_end - utterannce_start
                output_npz[target_seg_key] = input_npz[
                    utterance_key
                    ][start:end]
                n_target_segs += 1
                break

    print(
        "Extracted " + str(n_target_segs) + " out of " + str(len(target_segs))
        + " segments."
        )
    print("Writing:", output_npz_fn)
    np.savez(output_npz_fn, **output_npz)
