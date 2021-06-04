import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.data.dataset import SimpleDataset


class QADataset(SimpleDataset):
    '''A Dataset wrapping the TREC QA data.'''
    def __init__(self, filename, encoding='utf-8', subclasses=False):
        self._filename = filename
        self._encoding = encoding
        self._subclasses = subclasses
        super(QADataset, self).__init__(self._read())

    def _read(self):
        '''Reads each line of the input file and appends it to a list
        of question, label pairs.'''
        data = []
        with open(self._filename, 'r', encoding=self._encoding) as f:
            for line in f:
                label, question = line.split(' ', 1)
                if not self._subclasses:
                    label, _ = label.split(':')
                data.append((question, label))
        return data


    
class TextTransform(object):
    """
    Parameters:
        tokenizer (obj): Callable object that splits a string into a list of tokens
        vocabulary (:class:`gluonnlp.data.Vocab`): GluonNLP vocabulary
        pad (bool): Whether to pad data to maximum length
        max_seq_len (int): Maximum sequence/text length
    """
    def __init__(self,
                 tokenizer,
                 vocabulary,
                 pad=True,
                 max_seq_len=30):
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary
        self._max_seq_len = max_seq_len
        self._pad = pad

    def __call__(self, txt):
        """
        Parameters:
            txt (str): Input string representing a text document/passage
        Returns:
            (tuple): Tuple of:
                np.array - token id sequence
                np.array - array with single int for length of non-padded content
                np.array - mask with 1s for non-padded content and 0s for padded
        """
        tokens = self._tokenizer(txt)
        number_of_tokens = len(tokens)
        if number_of_tokens > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
            number_of_tokens = len(tokens)
        non_padded = np.array([number_of_tokens])
        if self._pad and number_of_tokens < self._max_seq_len:
            ids = np.zeros(self._max_seq_len, dtype=int)
            # pad with '<pad>' token
            tokens.extend(['<pad>'] * (self._max_seq_len - number_of_tokens))
            # get vocabulary index for each token
            for (i, token) in enumerate(tokens):
                ids[i] = self._vocabulary[token]
            # create mask with 1s for each element from the input text, and 0s for padding
            mask = np.array([1.0] * number_of_tokens + [0.0] * (self._max_seq_len - number_of_tokens))
        else:
            ids = np.zeros(number_of_tokens, dtype=int)
            # get voc index for each token
            for (i, token) in enumerate(tokens):
                ids[i] = self._vocabulary[token]
            # mask is all ones
            mask = np.ones(number_of_tokens, dtype=float)
        return (ids, non_padded, mask)


class ClassifierTransform(object):
    """
    Parameters:
        tokenizer (obj): Callable object that splits a string into a list of tokens
        vocabulary (:class:`gluonnlp.data.Vocab`): GluonNLP vocabulary
        max_seq_len (int): Maximum sequence/text length
        min_seq_len (int): Minimum sequence/text length
        pad (bool): Whether to pad data to maximum length
        class_labels (list): List of strings for the class labels
    """
    def __init__(self,
                 tokenizer,
                 vocabulary,
                 max_seq_len,
                 pad=True,
                 class_labels=None):
        self._text_xform = TextTransform(tokenizer, vocabulary, pad, max_seq_len)
        self._class_labels = class_labels
        self._label_map = {}
        for (i, label) in enumerate(class_labels):
            self._label_map[label] = i

    def __call__(self, labeled_text):
        """
        Parameters:
            labeled_text: tuple of str
                Input instances of (text, label) pairs
        Returns:
            np.array: token ids, shape (seq_length,)
            np.array: valid length, shape (1,)
            np.array: mask, shape (seq_length,)
            np.array: label id, shape (1,)
        """
        question, label_name = labeled_text
        # utilize the TextTransform object to get 3 of the arrays
        ids, num_tokens, mask = self._text_xform(question)
        # additional array to hold the label id
        label = np.array([self._label_map[label_name]], dtype='float32')
        return ids, num_tokens, mask, label


def build_vocabulary(dataset, tokenizer):
    """
    Parameters:
        dataset (:class:`QADataset`): QADataset to build vocab over
        tokenizer (obj): Callable object to split strings into tokens
    Returns:
        (:class:`gluonnlp.data.Vocab`): GluonNLP vocab object
    """
    all_tokens = []
    for (txt, label)  in dataset:
        all_tokens.extend(tokenizer(txt))
    counter = nlp.data.count_tokens(all_tokens)
    # init a Vocab object, create default unknown and padding tokens.
    vocab = nlp.Vocab(counter)
    return vocab


class BasicTokenizer(object):
    """Callable object to split strings into lists of tokens
    """
    def __call__(self, txt):
        return txt.split(' ')


def get_data_loaders(class_labels, train_file, dev_file, test_file, batch_size, max_len, pad):
    """
    Returns a vocabulary object and data loaders for train, dev, and test sets.
    """
    tokenizer = BasicTokenizer()
    train_ds = QADataset(train_file)
    print("Building Vocabulary", flush=True)
    vocabulary = build_vocabulary(train_ds, tokenizer)
    transform = ClassifierTransform(tokenizer, vocabulary, max_len, pad=pad, class_labels=class_labels)
    data_train = mx.gluon.data.SimpleDataset(list(map(transform, train_ds)))
    print("Getting DataLoaders", flush=True)
    data_train_lengths = data_train.transform(
        lambda ids, lengths, mask, label_id: lengths, lazy = False)
    # setup batches using fixed sized buckets
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, pad_val=0),
                                          nlp.data.batchify.Stack(),
                                          nlp.data.batchify.Pad(axis=0, pad_val=0),
                                          nlp.data.batchify.Stack())
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_lengths,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=4,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    if dev_file:
        dev_ds = QADataset(dev_file)
        data_dev = mx.gluon.data.SimpleDataset(list(map(transform, dev_ds)))
        loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)
    else:
        loader_dev = None
    if test_file:
        test_ds = QADataset(test_file)
        data_test = mx.gluon.data.SimpleDataset(list(map(transform, test_ds)))
        loader_test = mx.gluon.data.DataLoader(
            data_test,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)
    else:
        loader_test = None
    return vocabulary, loader_train, loader_dev, loader_test


def get_data_loaders_cross(class_labels, train_file, batch_size, max_len, pad, folds):
    """
    Returns a list of (Vocabulary, train loader, dev loader) tuples, one for each fold
    """
    tokenizer = BasicTokenizer()
    train_ds = QADataset(train_file)
    chunk_size = len(train_ds) // folds
    datasets = []
    # for each fold in the k-fold validation
    print("Getting DataLoaders", flush=True)
    for i in range(0, folds):
        # split the data into test and dev chunks
        dev_start_index = chunk_size * i
        if i == 0:
            dev_set = train_ds[:chunk_size]
            train_set = train_ds[chunk_size:]
        elif i == folds - 1:
            dev_set = train_ds[dev_start_index:]
            train_set = train_ds[:dev_start_index]
        else:
            dev_set = train_ds[dev_start_index:dev_start_index+chunk_size]
            train_set = train_ds[:dev_start_index] + train_ds[dev_start_index+chunk_size:]
        # build a vocabulary only from the training set
        vocabulary = build_vocabulary(train_set, tokenizer)
        transform = ClassifierTransform(tokenizer, vocabulary, max_len, pad=pad, class_labels=class_labels)
        # get the training dataset
        data_train = mx.gluon.data.SimpleDataset(list(map(transform, train_set)))
        data_train_lengths = data_train.transform(
            lambda ids, lengths, mask, label_id: lengths, lazy=False)
        batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, pad_val=0),
                                              nlp.data.batchify.Stack(),
                                              nlp.data.batchify.Pad(axis=0, pad_val=0),
                                              nlp.data.batchify.Stack())
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            data_train_lengths,
            batch_size=batch_size,
            num_buckets=10,
            ratio=0,
            shuffle=True)
        # and split it into batches
        loader_train = gluon.data.DataLoader(
            dataset=data_train,
            num_workers=4,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn)
        # get the dataset for the dev data
        data_dev = mx.gluon.data.SimpleDataset(list(map(transform, dev_set)))
        loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)
        # append the vocabulary, train data, dev data to the list of datasets per fold
        datasets.append((vocabulary, loader_train, loader_dev))
    return datasets