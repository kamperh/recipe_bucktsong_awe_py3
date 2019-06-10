Acoustic Word Embedding Models and Evaluation
=============================================

Data preparation
----------------
Create links to the MFCC NumPy archives:

    ./link_buckeye_mfcc.py
    ./link_xitsonga_mfcc.py
    
For Xitsonga, only UTD segments and test data is used; all validation (i.e.
choosing hyper-parameters) is based on the Buckeye English validation data.

The `gt2` set for Buckeye contains more word segments based on slightly relaxed
thresholds (see [features/readme.md](features/readme.md)). The number of tokens
in this set is similar to that in the `utd` set. However, since `gt2` contains
tokens that are less similar to tokens in `val` or `test`, using this set
actually results in worse performance for some of the models.


Correspondence autoencoder
--------------------------
Train an CAE-RNN on ground truth segments:

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

Train an CAE-RNN on UTD segments:

    # Buckeye
    ./train_cae.py --train_tag utd

    # Xitsonga
    ./train_cae.py --data_dir data/xitsonga.mfcc --train_tag utd \
        --pretrain_usefinal --extrinsic_usefinal --use_test_for_val
        --cae_n_epochs 3

Apply a Buckeye CAE-RNN on Xitsonga:

    ./apply_model.py --language xitsonga \
        models/buckeye.mfcc.gt/train_cae/546fd9ac51/cae.best_val.ckpt test
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_cae/546fd9ac51/cae.best_val.xitsonga.test.npz


Autoencoder
-----------
To train an AE-RNN, we actually use the same script as for the CAE-RNN, but
only with the AE pre-training step.

Train an AE-RNN on ground truth segments:

    ./train_cae.py --train_tag gt2 --cae_n_epochs 0

Train an AE-RNN on random segments:

    ./train_cae.py --train_tag rnd --cae_n_epochs 0

Train a AE-RNN on UTD segments:

    ./train_cae.py --train_tag utd --cae_n_epochs 0


Variational autoencoder
-----------------------
Train an VAE-RNN on ground truth segments:

    ./train_vae.py --train_tag gt

Train an VAE-RNN on random segments:

    ./train_vae.py --train_tag rnd

Train an VAE-RNN on UTD segments:

    ./train_vae.py --train_tag utd


Siamese model
-------------
Train a Siamese model on ground truth segments:

    ./train_siamese.py --n_epochs 50 --train_tag gt

Evaluate the model:

    ./apply_model.py \
        models/buckeye.mfcc.gt/train_siamese/c95c82710c/siamese.best_val.ckpt val
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt/train_siamese/c95c82710c/siamese.best_val.val.npz

This gives the results:

    Average precision: ?? # TO-DO
    Precision-recall breakeven: ?? # TO-DO


Siamese CNN
-----------
Train a Siamese CNN on ground truth segments:

    ./train_siamese_cnn.py --n_epochs 150 --train_tag gt2 --n_val_interval 5

Evaluate the model:

    ./apply_model.py \
        models/buckeye.mfcc.gt2/train_siamese_cnn/85116c3501/siamese_cnn.best_val.ckpt val
    ./eval_samediff.py --mvn \
        models/buckeye.mfcc.gt2/train_siamese_cnn/85116c3501/siamese_cnn.best_val.val.npz

This gives the results:

    Average precision: 0.6418
    Precision-recall breakeven: 0.6131


Sweeping across models
----------------------
Multiple models can be run in series and the evaluated. Here is an example of
running a model with different seeds:

    ./sweep.py --static_args "--train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.sweep1

The log files produced by `sweep.py` can be analysed as follows (this includes
combining the different evaluations):

    ./analyse_sweep.py models/train_cae.sweep1

Perform test set evaluation on a sweep:

    ./test_sweep.py models/train_cae.sweep1

After running `test_sweep.py`, `analyse_sweep.py` can be run again to also
include the test set analysis.


Results: Buckeye
----------------

### CAE-RNN trained on ground truth segments:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 30 --train_tag gt --pretrain_usefinal" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep1

Validation results:

    Validation AP mean: 0.4893 (+- 0.0119)
    Validation AP with normalisation mean: 0.4894 (+- 0.0145)

Test results (need to run `test_sweep.py` and then `analyse_sweep.py`, as
explained above):

    Test AP mean: 0.5212 (+- 0.0081)
    Test AP with normalisation mean: 0.5245 (+- 0.0113)

### CAE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 10 --train_tag utd --pretrain_usefinal" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep2

Results:

    Validation AP mean: 0.2979 (+- 0.0133)
    Validation AP with normalisation mean: 0.3262 (+- 0.0048)
    Test AP mean: 0.2922 (+- 0.0145)
    Test AP with normalisation mean: 0.3268 (+- 0.0052)

### CAE-RNN trained without initialising from AE-RNN:

Sweep:

    ./sweep.py --static_args \
        "--cae_n_epochs 150 --train_tag gt --ae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep3
    ./sweep.py --static_args \
        "--cae_n_epochs 150 --train_tag utd --ae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae.paper.sweep4

Results (ground truth):

    Validation AP mean: 0.4381 (+- 0.0079)
    Validation AP with normalisation mean: 0.4443 (+- 0.0049)

Results (UTD):

    Validation AP mean: 0.1436 (+- 0.0035)
    Validation AP with normalisation mean: 0.1730 (+- 0.0094)

### AE-RNN trained on ground truth segments:

Sweep:

    ./sweep.py --static_args "--train_tag gt --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep1

Validation results:

    Validation AP mean: 0.2521 (+- 0.0193)
    Validation AP with normalisation mean: 0.2688 (+- 0.0037)
    Test AP mean: 0.2297 (+- 0.0240)
    Test AP with normalisation mean: 0.2549 (+- 0.0050)

### AE-RNN trained on random segments:

Sweep:

    ./sweep.py --static_args "--train_tag rnd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep2

Results:

    Validation AP mean: 0.2591 (+- 0.0032)
    Validation AP with normalisation mean: 0.2687 (+- 0.0025)
    Test AP mean: 0.2430 (+- 0.0039)
    Test AP with normalisation mean: 0.2516 (+- 0.0018)

### AE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args "--train_tag utd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae.paper.sweep3  # HERE

Results:

    Validation AP mean: 0.2404 (+- 0.0074)
    Validation AP with normalisation mean: 0.2681 (+- 0.0031)
    Test AP mean: 0.2327 (+- 0.0065)
    Test AP with normalisation mean: 0.2551 (+- 0.0034)

### VAE-RNN trained on ground truth segments:

Sweep:

    ./sweep.py --static_args "--train_tag gt2 --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep1

Results:

    Validation AP mean: 0.2639 (+- 0.0023)
    Validation AP with normalisation mean: 0.2563 (+- 0.0038)

### VAE-RNN trained on random segments:

Sweep:

    ./sweep.py --static_args "--train_tag rnd --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep2

Results:

    Validation AP mean: 0.2562 (+- 0.0032)
    Validation AP with normalisation mean: 0.2619 (+- 0.0050)

### VAE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args "--train_tag utd --n_epochs 400" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_vae.paper.sweep3

Results:

    Validation AP mean: 0.2588 (+- 0.0031)
    Validation AP with normalisation mean: 0.2668 (+- 0.0033)
    Test AP mean: 0.2389 (+- 0.0034)
    Test AP with normalisation mean: 0.2506 (+- 0.0035)


Results: Xitsonga
-----------------

### CAE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --pretrain_usefinal --extrinsic_usefinal --use_test_for_val --cae_n_epochs 3 --train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_cae_xitsonga.paper.sweep1

Test results:

    Test AP mean: 0.2914 (+- 0.0140)
    Test AP with normalisation mean: 0.3242 (+- 0.0078)

Although it says "validation" in the output, remember that we used the test
data as validation data during training without cheating, i.e. we always used
the final model through `--extrinsic_usefinal`. This is also true for all the
results below.

### AE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --pretrain_usefinal --extrinsic_usefinal --use_test_for_val --train_tag utd --cae_n_epochs 0" \
        --rnd_seed 1,2,3,4,5 train_cae &> models/train_ae_xitsonga.paper.sweep2

Test results:

    Test AP mean: 0.1244 (+- 0.0102)
    Test AP with normalisation mean: 0.1389 (+- 0.0034)

### VAE-RNN trained on UTD segments:

Sweep:

    ./sweep.py --static_args \
        "--data_dir data/xitsonga.mfcc --n_epochs 300 --extrinsic_usefinal --use_test_for_val --train_tag utd" \
        --rnd_seed 1,2,3,4,5 train_vae &> models/train_va_xitsongae.paper.sweep3

Test results:

    Validation AP mean: 0.1175 (+- 0.0039)
    Validation AP with normalisation mean: 0.1158 (+- 0.0027)
