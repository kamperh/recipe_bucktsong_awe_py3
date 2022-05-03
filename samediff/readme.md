Same-Different Evaluation
=========================


Overview
--------
Performs same-different evaluation on frame-level features using dynamic time
warping (DTW) alignment.


Feature format
--------------
The input feature files are NumPy archives. Each key in the archive should have
the format `<label>_<speaker>_<other_identifiers>`. For instance
`acceptable_s19_01b_052667-052721` is the word "acceptable" spoken by speaker
"s19" in the Buckeye corpus, with the remaining part of the key indicating the
particular utterance and interval (but this is ignored). The label is used for
the word identity in the same-different task. The entry in this archive would
be the speech features (features are stacked row-wise). For instance
`feats["acceptable_s19_01b_052667-052721"].shape` whould be `(54, 39)`,
indicating 54 frames, each of 39 dimensions.


Evaluation
----------
This needs to be run on a multi-core machine. Change the `n_cpus` variable in
`run_calcdists.sh` and `run_samediff.sh` to the number of CPUs on the machine.

Evaluate MFCCs:

    # Devpart2
    ./run_calcdists.sh ../features/mfcc/buckeye/devpart2.samediff.dd.npz
    ./run_samediff.sh ../features/mfcc/buckeye/devpart2.samediff.dd.npz

    # ZeroSpeech
    ./run_calcdists.sh ../features/mfcc/buckeye/zs.samediff.dd.npz
    ./run_samediff.sh ../features/mfcc/buckeye/zs.samediff.dd.npz

    # Xitsonga
    ./run_calcdists.sh ../features/mfcc/xitsonga/xitsonga.samediff.dd.npz
    ./run_samediff.sh ../features/mfcc/xitsonga/xitsonga.samediff.dd.npz

Evaluate filterbanks:

    # Devpart2
    ./run_calcdists.sh ../features/fbank/buckeye/devpart2.samediff.npz
    ./run_samediff.sh ../features/fbank/buckeye/devpart2.samediff.npz

    # ZeroSpeech
    ./run_calcdists.sh ../features/fbank/buckeye/zs.samediff.npz
    ./run_samediff.sh ../features/fbank/buckeye/zs.samediff.npz

    # Xitsonga
    ./run_calcdists.sh ../features/fbank/xitsonga/xitsonga.samediff.npz
    ./run_samediff.sh ../features/fbank/xitsonga/xitsonga.samediff.npz


Results
-------
Devpart2 MFFCs:

    Average precision: 0.3758481108
    Precision-recall breakeven: 0.404233350542

Devpart2 filterbanks:

    Average precision: 0.192112817171
    Precision-recall breakeven: 0.256582343831

ZeroSpeech MFFCs:

    Average precision: 0.389555665546
    Precision-recall breakeven: 0.423245339672

ZeroSpeech filterbanks:

    Average precision: 0.19343299396
    Precision-recall breakeven: 0.262087947482
    
Xitsonga MFFCs:

    Average precision: 0.270764471475
    Precision-recall breakeven: 0.337214482423

Xitsonga filterbanks:

    Average precision: 0.123965806696
    Precision-recall breakeven: 0.205673902479
