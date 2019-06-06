Acoustic Word Embedding Models and Evaluation
=============================================

Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_buckeye_mfcc.py
    ./link_xitsonga_mfcc.py
    
For Xitsonga, only UTD segments and test data is used; all validation (i.e.
choosing hyper-parameters) is based on the Buckeye English validation data.


Correspondence autoencoder
--------------------------
Train an RNN-CAE on ground truth segments:

    # Buckeye
    ./train_cae.py --cae_n_epochs 30 --train_tag gt

Evaluate the model:

    # Validation
    ./apply_model.py \
        models/buckeye.mfcc.gt/train_cae/60a4a9619e/cae.best_val.ckpt val
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_cae/60a4a9619e/cae.best_val.val.npz

    # Test
    ./apply_model.py \
        models/buckeye.mfcc.gt/train_cae/60a4a9619e/cae.best_val.ckpt test
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_cae/60a4a9619e/cae.best_val.test.npz

All the models trained below (including autoencoder and VAE models) can be
evaluated using these scripts.

Analyse embeddings:

    ./analyse_embeds.py --normalise --word_type \
        because,yknow,people,something,anything,education,situation \
        models/buckeye.mfcc.utd/train_cae/e6f4584e05/cae.best_val.test.npz

    ./analyse_embeds.py --normalise --word_type \
        because,basically,exactly,probably,yknow,school,sometimes,something,education,situation \
        models/buckeye.mfcc.utd/train_cae/e6f4584e05/cae.best_val.test.npz

Train an RNN-CAE on UTD segments:

    # Buckeye
    ./train_cae.py --train_tag utd

    # Xitsonga
    ./train_cae.py --data_dir data/xitsonga.mfcc --train_tag utd \
        --pretrain_usefinal --extrinsic_usefinal --use_test_for_val
        --cae_n_epochs 3

Apply a Buckeye RNN-CAE on Xitsonga:

    ./apply_model.py --language xitsonga \
        models/buckeye.mfcc.gt/train_cae/546fd9ac51/cae.best_val.ckpt test
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_cae/546fd9ac51/cae.best_val.xitsonga.test.npz


Autoencoder
-----------
To train an RNN-AE, we actually use the same script as for the RNN-CAE, but
only with the AE pre-training step.

Train an RNN-AE on ground truth segments:

    ./train_cae.py --train_tag gt --cae_n_epochs 0

Train an RNN-AE on random segments:

    ./train_cae.py --train_tag rnd --cae_n_epochs 0

Train a RNN-AE on UTD segments:

    ./train_cae.py --train_tag utd --cae_n_epochs 0


Variational autoencoder
-----------------------
Train an RNN-VAE on ground truth segments:

    ./train_vae.py --train_tag gt

Train an RNN-VAE on random segments:

    ./train_vae.py --train_tag rnd

Train an RNN-VAE on UTD segments:

    ./train_vae.py --train_tag utd


Siamese model
-------------
Train a Siamese model on ground segments:

    ./train_siamese.py --n_epochs 50 --train_tag gt

Evaluate the model:

    # Validation
    ./apply_model.py \
        models/buckeye.mfcc.gt/train_siamese/c95c82710c/siamese.best_val.ckpt val
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_siamese/c95c82710c/siamese.best_val.val.npz

This gives the results:

    Average precision: ?? # TO-DO
    Precision-recall breakeven: ?? # TO-DO


Sweeping across models
----------------------
Multiple models can be run in series and the evaluated. Here is an example of
running a model with different seeds:

    ./sweep.py --static_args "--train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.sweep1

The log files produced by `sweep.py` can be analysed as follows (this includes
combining the different evaluations):

    ./analyse_sweep.py models/train_cae.sweep1

Perform test-set evaluation on a sweep:

    ./test_sweep.py models/train_cae.sweep1


Results: Buckeye
----------------

### RNN-CAE trained on ground truth segments:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 30 --train_tag gt --pretrain_usefinal" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep1  # HERE

Validation results:

    Validation AP mean: 0.4893 (+- 0.0119)
    Validation AP with normalisation mean: 0.4894 (+- 0.0145)

Test results (needs to be performed manually on each model and saved in
`test_ap.txt` in the model directory):

    Test AP mean: 0.3450 (+- 0.0221)
    Test AP with normalisation mean: 0.3455 (+- 0.0208)

### RNN-CAE trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 10 --train_tag utd --pretrain_usefinal" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep2

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)
    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

### RNN-CAE trained without initialising from RNN-AE:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 150 --train_tag gt --ae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep3
    ./sweep.py --static_args \
        "--cae_n_epochs 150 --train_tag utd --ae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep4

Results (ground truth):

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)

Results (UTD):

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)

### RNN-AE trained on ground truth segments:

Sweep:

    ./sweep.py --static_args "--train_tag gt --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep1

Validation results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)
    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

### RNN-AE trained on random segments:

Sweep:

    ./sweep.py --static_args "--train_tag rnd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep2

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)
    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

### RNN-AE trained on UTD segments:

Sweep:

    ./sweep.py --static_args "--train_tag utd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep5

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)
    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

### RNN-VAE trained on ground truth segments:

Sweep:

    ./sweep.py --static_args "--train_tag gt --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep10

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)

### RNN-VAE trained on random segments:

Sweep:

    ./sweep.py --static_args "--train_tag rnd --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep9

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)

### RNN-VAE trained on UTD segments:

Sweep:

    ./sweep.py --static_args "--train_tag utd --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep8

Results:

    Validation AP mean: ?? (+- ??)
    Validation AP with normalisation mean: ?? (+- ??)
    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)


Results: Xitsonga
-----------------

### RNN-CAE trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --pretrain_usefinal --extrinsic_usefinal --use_test_for_val --cae_n_epochs 3 --train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae_xitsonga.paper.sweep8

Test results:

    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

Although it says "validation" in the output, remember that we used the test
data as validation data during training without cheating, i.e. we always used
the final model through `--extrinsic_usefinal`. This is also true for all the
results below.

### RNN-AE trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --pretrain_usefinal --extrinsic_usefinal --use_test_for_val --train_tag utd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae_xitsonga.paper.sweep9

Test results:

    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)

### RNN-VAE trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --n_epochs 300 --extrinsic_usefinal --use_test_for_val --train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_va_xitsongae.paper.sweep11

Test results:

    Test AP mean: ?? (+- ??)
    Test AP with normalisation mean: ?? (+- ??)
