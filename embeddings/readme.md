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


