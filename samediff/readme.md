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



    # Xitsonga  TO-DO
    ./run_calcdists.sh \
        ../features/wordpairs/xitsonga/xitsonga.samediff.mfcc.cmvn_dd.npz
    ./run_samediff.sh  \
        ../features/wordpairs/xitsonga/xitsonga.samediff.mfcc.cmvn_dd.npz

Evaluate filterbanks:

    # Devpart2
    ./run_calcdists.sh ../features/fbank/buckeye/devpart2.samediff.npz
    ./run_samediff.sh ../features/fbank/buckeye/devpart2.samediff.npz




    ./run_calcdists.sh \
        ../features/wordpairs/devpart2/devpart2.samediff.fbank.mvn.npz
    ./run_samediff.sh  \
        ../features/wordpairs/devpart2/devpart2.samediff.fbank.mvn.npz

    # Xitsonga
    ./run_calcdists.sh \
        ../features/wordpairs/xitsonga/xitsonga.samediff.fbank.mvn.npz
    ./run_samediff.sh  \
        ../features/wordpairs/xitsonga/xitsonga.samediff.fbank.mvn.npz


Results
-------
Devpart2 MFFCs:

    Average precision: 0.369589218033
    Precision-recall breakeven: 0.397548380274

Devpart2 filterbanks:

    # TO-DO
    Average precision: 0.19268681067
    Precision-recall breakeven: 0.256066081569

ZeroSpeech MFFCs:

    Average precision: 0.360448500146
    Precision-recall breakeven: 0.396221693982
    
Xitsonga MFFCs:

    # TO-DO
    Average precision: 0.281450179468
    Precision-recall breakeven: 0.344160051839

Xitsonga filterbanks:

    # TO-DO
    Average precision: 0.127098865168
    Precision-recall breakeven: 0.207577352989
