# Model-Chaining-for-Question-Classification

This project is an investigation into the effectiveness of chaining CNNs and LSTMs for the task of question classification. See the [write up](https://github.com/krajovic/Model-Chaining-for-Question-Classification/blob/main/WriteUp.pdf) for more information on model design and experimentation. The code can train CNN, LSTM, CNN-LSTM, and LSTM-CNN models. The model can be specified using the `--lstm` `--cnn_lstm` `--lstm_cnn` flags (the CNN model is the default). 

Run `$ ./train.py -h` for a list of adjustable parameters. 
