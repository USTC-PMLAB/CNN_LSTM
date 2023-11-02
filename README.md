# CNN_LSTM
# **Deep learning-based nanomechanical vibration for rapid and label-free assay of epithelial mesenchymal** **transition**

###### CNN-LSTM is the source code for "**Deep learning-based nanomechanical vibration for rapid and label-free assay of epithelial mesenchymal** **transition**"

It contains CNN-like architecture for feature extraction and LSTM-like architecture for processing long time-series data.

### Requirements 

##### Requirements

tensorflow version = 2.10.0

keras version = 2.10.0

numpy version >= 1.21.5

matplotlib>= 0.1.6

### Datasets

###### Training sets are placed in dl_data_train.csv and test sets are placed in dl_data_test.csv.

In training sets and test sets, the first row represents the label of each data. Label 0 refers to controlled group. Label 1, 2, 3 represent 24h, 48h, 72h, respectively.

### Get Started!

##### Training from Scratch

`python cnn2_lstm_emt_four_sort.py`

##### Test for Validation

`python cnn2_lstm_test.py`
