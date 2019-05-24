Same-Different Evaluation
=========================


Overview
--------
Performs same-different evaluation on frame-level features using dynamic time
warping (DTW) alignment.


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

    Average precision: 0.369589218033
    Precision-recall breakeven: 0.397548380274

Devpart2 filterbanks:

    Average precision: 0.192686800537
    Precision-recall breakeven: 0.256066081569

ZeroSpeech MFFCs:

    Average precision: 0.360448500146
    Precision-recall breakeven: 0.396221693982

ZeroSpeech filterbanks:

    Average precision: 0.193150835418
    Precision-recall breakeven: 0.261508627742
    
Xitsonga MFFCs:

    Average precision: 0.265691186512
    Precision-recall breakeven: 0.329374601376

Xitsonga filterbanks:

    Average precision: 0.12477363302
    Precision-recall breakeven: 0.20666915823
