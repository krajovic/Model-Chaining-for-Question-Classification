#!/usr/bin/env python3

import argparse
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluonnlp as nlp
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

from load_data import get_data_loaders, get_data_loaders_cross
from model import CNN, LSTM, LSTMCNN, CNNLSTM


LOSS_FUNC = gluon.loss.SoftmaxCrossEntropyLoss()
CLASS_LABELS = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']


def get_model(emb_input_dim, emb_output_dim, num_classes):
    '''Returns the appropriate model according to the
    specified keyword arguments'''
    if args.lstm:
        return LSTM(
            emb_input_dim,
            emb_output_dim,
            num_classes=num_classes,
            dr=args.dropout
        )
    else:
        # convert input list of filter sizes
        filters = [int(x) for x in args.filter_sizes.split(',')]
        if args.lstm_cnn:
            return LSTMCNN(
                emb_input_dim,
                emb_output_dim,
                num_classes=num_classes,
                dr=args.dropout,
                filter_widths=filters,
                num_filters=args.num_filters
            )
        elif args.cnn_lstm:
            return CNNLSTM(
                emb_input_dim,
                emb_output_dim,
                num_classes=num_classes,
                dr=args.dropout,
                filter_widths=filters,
                num_filters=args.num_filters
            )
        else:   # if nothing is specified assume CNN
            return CNN(
                emb_input_dim,
                emb_output_dim,
                num_classes=num_classes,
                dr=args.dropout,
                filter_widths=filters,
                num_filters=args.num_filters
            )

def cross_validate(datasets, ctx=mx.cpu()):
    '''Perform k-fold cross validation on the train set.
    The input list of datasets has elements equal to the
    number of folds. Each contains a vocabulary, training
    set and a dev set.'''
    print(f"{len(datasets)}-fold cross validation:")
    dev_accs = []
    for i, fold in enumerate(datasets):
        print(f"------------------- FOLD {i+1} -------------------")
        vocab, train, dev = fold
        # set up embeddings for vocabulary
        glove = nlp.embedding.create('glove', source=args.embedding_source)
        vocab.set_embedding(glove)
        emb_size = vocab.embedding.idx_to_vec.shape[1]
        oov_items = 0
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                oov_items += 1
                vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)
        # train the classifier
        fold_acc = train_classifier(vocab, train, dev, None, emb_size, ctx)
        # store the dev accuracy
        dev_accs.append(fold_acc)
    # log the results
    print("======================== TRAINING COMPLETED ========================")
    print(f"Dev accuracies: {dev_accs}")
    avg_acc = sum(dev_accs)/len(dev_accs)
    print(f"AVERAGE ACCURACY: {avg_acc}")



def train_classifier(vocabulary, train_loader, val_loader, test_loader, embedding_size, ctx=mx.cpu()):
    '''Trains the model'''
    emb_input_dim, emb_output_dim = len(vocabulary.idx_to_token), embedding_size
    num_classes = len(CLASS_LABELS)
    model = get_model(emb_input_dim, emb_output_dim, num_classes)
    # initialize model parameters on the context ctx
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)
    # set the embedding layer with the vocabulary and pretrained embeddings
    model.embedding_layer.weight.set_data(vocabulary.embedding.idx_to_vec)
    # initialize a Trainer passing in the learning rate and weight decay factor
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr, 'wd': args.wd})
    # perform an initial baseline evaluation
    start_ap, start_acc = evaluate(model, val_loader, verbose=args.verbose)
    print(f"Starting AP = {start_ap} Acc = {start_acc}")
    max_val_acc = 0
    max_val_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print(f"EPOCH: {epoch}")
        epoch_loss = 0
        for i, (ids, lens, mask, label) in enumerate(train_loader):
            ids = ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            with autograd.record():
                output = model(ids, mask)
                loss = LOSS_FUNC(output, label)
                loss = loss.mean()
            loss.backward()
            trainer.step(1, ignore_stale_grad=True)
            epoch_loss += loss.asscalar()
        print(f"Epoch loss = {epoch_loss}")
        tr_ap, tr_acc = evaluate(model, train_loader, verbose=args.verbose)
        print(f"TRAINING AP = {tr_ap} Acc = {tr_acc}")
        val_ap, val_acc = evaluate(model, val_loader, verbose=args.verbose)
        # keep track of the maximum accuracy achieved on dev
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_epoch = epoch
        print(f"VALIDATION AP = {val_ap} Acc = {val_acc}")
    # log the max accuracy and epoch on which it was achieved
    print(f'MAX VAL ACC: {max_val_acc} EPOCH: {max_val_epoch}')
    # final evaluation on the test set
    if test_loader is not None:
        print("======================== TRAINING COMPLETED ========================")
        tst_ap, tst_acc = evaluate(model, test_loader, verbose=True)
        print(f"TEST AP = {tst_ap} Acc = {tst_acc}")
    return max_val_acc


def evaluate(model, dataloader, verbose=False):
    """Evaluates the model on val or test set."""
    total_correct = 0
    total = 0
    gold = None # the correct labels for whole dataset
    pred = None # predicted labels for whole dataset
    probs = None # the weights produced by the model for whole dataset
    for i, (ids, lens, mask, label) in enumerate(dataloader):
        out = model(ids, mask)
        # predictions are argmax
        y_hat = mx.nd.argmax(out, axis=1)
        y = label.squeeze()
        # for each of the three values we're tracking, append this batch to overall
        if gold is not None:
            gold = mx.nd.concat(gold, y, dim=0)
        else:
            gold = y
        if pred is not None:
            pred = mx.nd.concat(pred, y_hat, dim=0)
        else:
            pred = y_hat
        if probs is not None:
            probs = mx.nd.concat(probs, out, dim=0)
        else:
            probs = out
        # calculate total correct and incremement for full dataset
        total_correct += (mx.nd.argmax(out, axis=1) == label.squeeze()).sum().asscalar()
        total += label.shape[0]
    acc = total_correct / float(total)
    # convert each nd array to np array
    gold = gold.asnumpy()
    pred = pred.asnumpy()
    probs = probs.asnumpy()
    # binarize the labels to work with sklearn average_precision_score
    # label for each of the 6 classes
    bin_labels = label_binarize(gold, classes=[0, 1, 2, 3, 4, 5])
    ap = average_precision_score(bin_labels, probs)
    # if we want many statistics, get precision, recall, fbeta for each class
    if verbose:
        prf = precision_recall_fscore_support(gold, pred, zero_division=0)
        met = ['Precision: ', 'Recall: ', 'FBeta: ', 'Samples: ']
        for i, label in enumerate(prf):
            metric_string = ""
            for j, label_name in enumerate(CLASS_LABELS):
                metric_string = " ".join([metric_string, f"{label_name}-{label[j]}"])
            print(" ".join([met[i], metric_string]))
    return ap, acc


def model_specs():
    '''Prints model parameters.'''
    if args.lstm:
        model = "LSTM"
    elif args.lstm_cnn:
        model = "LSTM-CNN"
    elif args.cnn_lstm:
        model = "CNN-LSTM"
    else:
        model = "CNN"
    print(f"Training {model} Model")
    if args.cross_val:
        print(f"{args.folds}-fold Cross Validation")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}, WD: {args.wd}, DR: {args.dropout}")
    if args.pad_data:
        print(f"Padded, Seq Length: {args.seq_length}, Batch Size: {args.batch_size}")
    else:
        print(f"Max Seq Length: {args.seq_length}, Batch Size: {args.batch_size}")
    print(f"Embeddings: {args.embedding_source}")
    if not args.lstm:
        print(f"Filter Sizes: {args.filter_sizes}, Num Filters: {args.num_filters}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a question classifier - via convolutional, recurrent, or chained architecture')
    parser.add_argument('--train_file', type=str, help='Path to file representing the input lTRAINING data', default=None)
    parser.add_argument('--val_file', type=str, help='Path to file representing the input VALIDATION data', default=None)
    parser.add_argument('--test_file', type=str, help='Path to file representing the input TEST data', default=None)
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--wd', type=float, help='Weight decay', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=32)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
    parser.add_argument('--seq_length', type=int, help='Max sequence length', default=30)
    parser.add_argument('--embedding_source', type=str, default='glove.6B.100d',
                        help='Pre-trained embedding source name (GluonNLP)')
    parser.add_argument('--lstm', action='store_true', help='Use an LSTM model', default=False)
    parser.add_argument('--lstm_cnn', action='store_true', help='Use the LSTM-CNN model', default=False)
    parser.add_argument('--cnn_lstm', action='store_true', help='Use the CNN-LSTM model', default=False)
    parser.add_argument('--filter_sizes', type=str, default='3,4', help='List of integer filter sizes (for CNN models only)')
    parser.add_argument('--num_filters', type=int, default=20, help='Number of filters (of each size, for CNN models only)')
    parser.add_argument('--pad_data', action='store_true', help='Explicitly pad all data to seq_length', default=False)
    parser.add_argument('--verbose', action='store_true', help='Print all stats on test set on each iteration', default=False)
    parser.add_argument('--cross_val', action='store_true', help='Perform k-fold cross validation', default=False)
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross validation')

    args = parser.parse_args()

    model_specs()

    ctx = mx.cpu()
    if args.cross_val:
        data_sets = get_data_loaders_cross(CLASS_LABELS,
                                           args.train_file,
                                           args.batch_size,
                                           args.seq_length,
                                           args.pad_data,
                                           args.folds)
        cross_validate(data_sets, ctx)
    else:
        vocab, train_loader, val_loader, test_loader = \
            get_data_loaders(CLASS_LABELS,
                             args.train_file,
                             args.val_file,
                             args.test_file,
                             args.batch_size,
                             args.seq_length,
                             args.pad_data)

        ## initialize pre-trained embeddings in Vocabulary
        glove = nlp.embedding.create('glove', source=args.embedding_source)
        ## use vocab.set_embedding to attach to the vocabulary
        vocab.set_embedding(glove)
        ## set embeddings to random for out of vocab items
        emb_size = vocab.embedding.idx_to_vec.shape[1]
        oov_items = 0
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                oov_items += 1
                vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)
        train_classifier(vocab, train_loader, val_loader, test_loader, emb_size, ctx)
