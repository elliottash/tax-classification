# U.S. Tax Law Classifier

This repository accompanies the paper 

* Luyang Han, Malka Guillot, and Elliott Ash, "Machine Extraction of Tax Laws from Legislative Texts", Workshop on Natural Legal Language Processing (2021).

The repo contains scripts and trained models for for identifying tax-related documents in legislative text, and further predicting the tax source of those documents (e.g. income tax, sales tax, property tax). 

We have replication code for training Logistic Regression and Random Forest models on the annotated statutes corpus. The script for training the classifier for "is tax related" is in the tax_source folder. The script for training the classifier for tax source is in the tax_source folder. 

The trained model can then be applied to other corpora. For example, in the paper, we apply it to historical statutes. The trained random forest models (with best performance from the paper) are available here:

* tax-related prediction: https://drive.google.com/file/d/1gUhbxmk213Z4PQugyr0UyBSVFBR1qkAv/view?usp=sharing
* tax-source prediction: https://drive.google.com/file/d/1X4dAHsPFabEE4D0HX9O2O1gbaBGBLlxt/view?usp=sharing

The sample scripts for using the trained models are included as `prediction_sample_statutes.py` and `prediction_sample_sentence_source.py`.
