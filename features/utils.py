"""
Utility functions

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

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
            utterance = utterance.replace("_", "-")
            utt_key = utterance[0:3] + "_" + utterance[3:]
            if utt_key not in vad_dict:
                vad_dict[utt_key] = []

            if token_label in ["SIL", "SPN"]:
                continue
            if prev_end_time != start_token or prev_utterance != utterance:
                if prev_end_time != -1:
                    utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
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

        utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
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
            utterance = utterance.replace("_", "-")
            start = float(start)
            end = float(end)
            if label in ["SIL", "SPN"]:
                continue
            words.append((utterance, label, (start, end)))

    print("Finding word tokens.")
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
            utterance = utterance[:3] + "_" + utterance[3:]
            f.write(
                label + "_" + utterance + "_%06d-%06d\n" % (int(start),
                int(end) + 1)
                )
