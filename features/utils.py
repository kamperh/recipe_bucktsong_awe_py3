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


def write_samediff_words(fa_fn, output_fn, min_frames=50, min_chars=5):
    """
    Find words of at least `min_frames` frames and `min_chars` characters.

    Ground truth words are extracted from the forced alignment file `fa_fn` and
    written to the word list file `output_fn`.
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
        if end_frame - start_frame >= min_frames and len(label) >= min_chars:
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
        + " segments"
        )
    print("Writing:", output_npz_fn)
    np.savez(output_npz_fn, **output_npz)


def strip_nonvad(utt, start, end, vads):
    """
    Return
    ------
    (nonvad_start, nonvad_end) : (int, int)
        Updated term indices. None is returned if the term does not fall in a
        VAD region.
    """

    # Get the VAD regions
    vad_starts = [i[0] for i in vads]
    vad_ends = [i[1] for i in vads]

    # Find VAD region with maximum overlap
    overlaps = []
    for (vad_start, vad_end) in zip(vad_starts, vad_ends):
        if vad_end <= start:
            overlaps.append(0)
        elif vad_start >= end:
            overlaps.append(0)
        else:
            overlap = end - start
            if vad_start > start:
                overlap -= vad_start - start
            if vad_end < end:
                overlap -= end - vad_end
            overlaps.append(overlap)
    
    if np.all(np.array(overlaps) == 0):
        # This term isn't in VAD.
        return None

    i_vad = np.argmax(overlaps)
    vad_start = vad_starts[i_vad]
    vad_end = vad_ends[i_vad]
    # print("VAD with max overlap:", (vad_start, vad_end))

    # Now strip non-VAD regions
    if vad_start > start:
        start = vad_start
    if vad_end < end:
        end = vad_end

    return (start, end)


def strip_nonvad_from_pairs(vad_dict, input_pairs_fn, output_pairs_fn,
        log=False):

    # Now keep only VAD regions
    if log:
        print("-"*39)
    f = open(output_pairs_fn, "w")
    print("Writing:", output_pairs_fn)
    for line in tqdm(open(input_pairs_fn)):

        line = line.strip().split(" ")

        if len(line) == 9:
            # Aren's format
            (
                cluster, utt1, speaker1, start1, end1, utt2, speaker2, start2,
                end2
            ) = line
            start1 = int(start1)
            end1 = int(end1)
            start2 = int(start2)
            end2 = int(end2)
            utt1 = uttlabel_to_uttkey(utt1)
            utt2 = uttlabel_to_uttkey(utt2)
            # utt1 = utt1.replace("_", "-")
            # utt2 = utt2.replace("_", "-")
        elif len(line) == 6:
            # Sameer's format
            utt1, start1, end1, utt2, start2, end2 = line
            speaker1 = utt1[:3]
            speaker2 = utt2[:3]
            cluster = "?"            
            start1 = int(np.floor(float(start1)*100))
            end1 = int(np.floor(float(end1)*100))
            start2 = int(np.floor(float(start2)*100))
            end2 = int(np.floor(float(end2)*100))

        # Utterances missing from forced alignments
        if utt1 == "s01_03a" or utt2 == "s01_03a":
            continue

        if (utt1 == utt2) and (start2 <= start1 <= end2 or
                    start2 <= end1 <= end2):
            if log:
                print(
                    "Warning: pairs from overlapping speech:", utt1, start1,
                    end1, start2, end2
                    )
            continue

        # Process the first term in the pair
        if log:
            print(
                "Processing:", utt1, "(" + str(start1) + ", " + str(end1) +
                "), cluster", cluster
                )
        if speaker1 == "nch":
            wav1_fn = utt1.replace("-", "_") + ".wav"
            wav2_fn = utt1.replace("-", "_") + ".wav"
        else:
            wav1_fn = "%s/%s.wav" % (speaker1, utt1.replace("_", ""))
            wav2_fn = "%s/%s.wav" % (speaker2, utt2.replace("_", ""))
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav1_fn,
            float(start1)/100., float(end1 - start1)/100.)
            )
        if log:
            print("Raw term play command:", sox_play_str)
        # if utt1 == "s05_03b":
        #     print(utt1, start1, end1, vad_dict[utt1])
        #     assert False
        nonvad_indices = strip_nonvad(utt1, start1, end1, vad_dict[utt1])
        if nonvad_indices is None:
            continue
        nonvad_start1, nonvad_end1 = nonvad_indices
        if nonvad_start1 != start1 or nonvad_end1 != end1:
            if log:
                print("Term changed")
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav1_fn,
            float(nonvad_start1)/100., float(nonvad_end1 - nonvad_start1)/100.)
            )
        if log:
            print("Term play command after VAD:", sox_play_str)

        if log:
            print()
            print(
                "Processing:", utt2, "(" + str(start2) + ", " +
                str(end2) + "), cluster", cluster
                )
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav2_fn,
            float(start2)/100., float(end2 - start2)/100.)
            )
        if log:
            print("Raw term play command:", sox_play_str)
        nonvad_indices = strip_nonvad(utt2, start2, end2, vad_dict[utt2])
        if nonvad_indices is None:
            continue
        nonvad_start2, nonvad_end2 = nonvad_indices
        if nonvad_start2 != start2 or nonvad_end2 != end2:
            if log:
                print("Term changed")
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav2_fn,
            float(nonvad_start2)/100., float(nonvad_end2 - nonvad_start2)/100.)
            )
        if log:
            print("Term play command after VAD:", sox_play_str)

        f.write(
            cluster + " " + utt1 + " " + str(nonvad_start1) + " " +
            str(nonvad_end1) + " " + utt2 + " " + str(nonvad_start2) + " " +
            str(nonvad_end2) + "\n"
            )

        if log:
            print("-"*39)
        # break
    if log:
        print("Wrote updated pairs:", output_pairs_fn)
    f.close()


def terms_from_pairs(pairs_fn, output_list_fn):

    print("Reading:", pairs_fn)
    terms = set()
    with open(pairs_fn) as f:
        for line in f:
            (
                cluster, utt1, start1, end1, utt2, start2, end2
            ) = line.strip().split(" ")
            start1 = int(start1)
            end1 = int(end1)
            start2 = int(start2)
            end2 = int(end2)
            terms.add((cluster, utt1, start1, end1))
            terms.add((cluster, utt2, start2, end2))

    print("Writing:", output_list_fn)
    with open(output_list_fn, "w") as f:
        for cluster, utt, start, end in terms:
            f.write(
                cluster + "_" + utt + "_" + "%06d" % start + "-" + "%06d" % end
                + "\n"
                )


def pairs_for_speakers(speakers_fn, input_pairs_fn, output_pairs_fn):

    print("Reading:", speakers_fn)
    with open(speakers_fn) as f:
        speakers = [line.strip() for line in f]

    print("Reading:", input_pairs_fn)
    print("Writing:", output_pairs_fn)
    input_pairs_f = open(input_pairs_fn)
    output_pairs_f = open(output_pairs_fn, "w")
    n_total = 0
    n_pairs = 0
    for line in input_pairs_f:
        _, utt1, _, _, utt2, _, _ = line.split()
        n_total += 1
        if utt1[:3] in speakers and utt2[:3] in speakers:
            output_pairs_f.write(line)
            n_pairs += 1
    input_pairs_f.close()
    output_pairs_f.close()
    
    print("Wrote", n_pairs, "out of", n_total, "pairs")

