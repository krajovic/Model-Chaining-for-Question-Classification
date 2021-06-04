import numpy as np
from mxnet import gluon
from mxnet.gluon import Block


class CNN(Block):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 6
        Number of categories in classifier output
    dr : float, default 0.5
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 20
        Number of filters for each width
    """

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=6, dr=0.5, filter_widths=[3, 4], num_filters=20):
        super().__init__()
        # embedding layer
        self.embedding_layer = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
        # HORIZONTAL CONCATENATION METHOD
        # parallel layers, one for each kernel width
        self.layers = gluon.contrib.nn.Concurrent()
        fs = [int(f) for f in filter_widths]
        for f in fs:
            self.layers.add(gluon.nn.Conv1D(num_filters, f, activation='relu'))
        # VERTICAL CONCATENATION METHOD
        # self.l1 = gluon.nn.Conv1D(num_filters, 2, padding=0, activation='relu')
        # self.l2 = gluon.nn.Conv1D(num_filters, 4, padding=1, activation='relu')
        # self.layers = gluon.contrib.nn.HybridConcurrent(axis=1)
        # self.layers.add(self.l1)
        # self.layers.add(self.l2)
        # global average pooling
        self.global_pooling = gluon.nn.GlobalAvgPool1D(layout='NCW')
        # drouput layer to help prevent overfitting
        self.dropout = gluon.nn.Dropout(dr)
        # final dense layer
        self.dense = gluon.nn.Dense(num_classes)

    def forward(self, data, mask):
        # first layer is embeddings
        embedding = self.embedding_layer(data)
        # configure mask to a useable shape
        mask = np.repeat(mask[:, :, np.newaxis], np.shape(embedding)[2], axis=2)
        # mask the input after embedding lookup
        x = embedding * mask.astype('float32')
        # transpose so that shapes work
        x = np.transpose(x, (0, 2, 1))
        # send through parallel convolutional layers
        x = self.layers(x)
        # perform global pooling
        x = self.global_pooling(x)
        # dropout for overfitting
        x = self.dropout(x)
        # final dense layer fo narrow to 6 classes
        x = self.dense(x)
        return x


class LSTM(Block):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 6
        Number of categories in classifier output
    dr : float, default 0.5
        Dropout rate
    """

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=6, dr=0.5):
        super().__init__()
        # embedding layer
        self.embedding_layer = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
        # encoder LSTM
        self.encoder = gluon.rnn.LSTM(emb_output_dim, dropout=dr, input_size=emb_output_dim, bidirectional=True)
        # pooling layer
        self.pooling_layer = gluon.nn.GlobalAvgPool1D(layout='NWC')
        # dense decoding layer
        self.decoder = gluon.nn.Dense(num_classes)

    def forward(self, data, mask):
        # embedding lookup
        embedding = self.embedding_layer(data)
        # configure mask to correct shape and mask input
        mask = np.repeat(mask[:, :, np.newaxis], np.shape(embedding)[2], axis=2)
        x = embedding * mask.astype('float32')
        # through LSTM
        x = self.encoder(x)
        # perform pooling
        x = self.pooling_layer(x)
        # narrow to 6 classes
        x = self.decoder(x)
        return x


class LSTMCNN(Block):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 6
        Number of categories in classifier output
    dr : float, default 0.5
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 20
        Number of filters for each width
    """
    def __init__(self, emb_input_dim, emb_output_dim, num_classes=6, dr=0.5, filter_widths=[3, 4], num_filters=20):
         super().__init__()
         # embedding lookup layer
         self.embedding_layer = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
         # LSTM encoder layer
         self.encoder = gluon.rnn.LSTM(emb_output_dim, dropout=dr, input_size=emb_output_dim, bidirectional=True)
         # HORIZONTAL CONCATENATION METHOD
         #self.layers = gluon.contrib.nn.Concurrent()
         #self.layers.add(gluon.nn.Conv1D(num_filters, 2, activation='relu'))
         #self.layers.add(gluon.nn.Conv1D(num_filters, 4, activation='relu'))
         # for f in filter_widths:
         #     self.layers.add(gluon.nn.Conv1D(num_filters, int(f), activation='relu'))
         # VERTICAL CONCATENATION METHOD
         self.l1 = gluon.nn.Conv1D(num_filters, 2, padding=0, activation='relu')
         self.l2 = gluon.nn.Conv1D(num_filters, 4, padding=1, activation='relu')
         self.layers = gluon.contrib.nn.Concurrent(axis=1)
         self.layers.add(self.l1)
         self.layers.add(self.l2)
         # global average pooling
         self.pooling_layer = gluon.nn.GlobalAvgPool1D(layout='NCW')
         # dropout to help with overfitting
         self.dropout = gluon.nn.Dropout(dr)
         # dense layer to narrow to 6 classes
         self.dense = gluon.nn.Dense(num_classes)


    def forward(self, data, mask):
         # embedding lookup
         embedding = self.embedding_layer(data)
         # configure mask to correct shape
         mask = np.repeat(mask[:, :, np.newaxis], np.shape(embedding)[2], axis=2)
         x = embedding * mask.astype('float32')
         # through LSTM
         x = self.encoder(x)
         # prepare for input to CNN
         x = np.transpose(x, (0, 2, 1))
         # send through all Convolutional layers
         x = self.layers(x)
         # perform global pooling
         x = self.pooling_layer(x)
         # dropout for overfitting
         x = self.dropout(x)
         # final dense layer fo narrow to 6 classes
         x = self.dense(x)
         return x


class CNNLSTM(Block):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 6
        Number of categories in classifier output
    dr : float, default 0.5
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 20
        Number of filters for each width
    """
    def __init__(self, emb_input_dim, emb_output_dim, num_classes=6, dr=0.5, filter_widths=[3, 4], num_filters=20):
        super().__init__()
        self.embedding_layer = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
        self.encoder = gluon.rnn.LSTM(num_filters*2, dropout=dr, input_size=num_filters*2, bidirectional=True)
        # two convolutional layers
        self.l1 = gluon.nn.Conv1D(num_filters, 2, padding=0, activation='relu')
        # pad this layer to make the output lengths match
        self.l2 = gluon.nn.Conv1D(num_filters, 4, padding=1, activation='relu')
        # parallel layers with vertical concatenation
        self.layers = gluon.contrib.nn.Concurrent(axis=1)
        self.layers.add(self.l1)
        self.layers.add(self.l2)
        # global average pooling
        self.pooling_layer = gluon.nn.GlobalAvgPool1D(layout='NWC')
        # dropout to prevent overfitting
        self.dropout = gluon.nn.Dropout(dr)
        # final dense layer to narrow to 6 classes
        self.dense = gluon.nn.Dense(num_classes)

    def forward(self, data, mask):
        # embedding lookup
        embedding = self.embedding_layer(data)
        # configure mask to correct shape
        mask = np.repeat(mask[:, :, np.newaxis], np.shape(embedding)[2], axis=2)
        x = embedding * mask.astype('float32')
        # prepare input for CNN
        x = np.transpose(x, (0, 2, 1))
        # run through convolutional layers
        x = self.layers(x)
        # prepare input for LSTM
        x = np.transpose(x, (0, 2, 1))
        # run through LSTM
        x = self.encoder(x)
        # perform pooling
        x = self.pooling_layer(x)
        # dropout for overfitting
        x = self.dropout(x)
        # final dense layer fo narrow to 6 classes
        x = self.dense(x)
        return x

