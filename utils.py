import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe

# load SST dataset
def sst(TEXT, LABEL, batch_size):
    """
    load SST (Stanford Sentiment Treebank) Dataset
    https://nlp.stanford.edu/sentiment/treebank.html
    """
    train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True)
    TEXT.build_vocab(train, val, test)
    LABEL.build_vocab(train, val, test)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                                        (train, val, test),
                                        batch_sizes=(batch_size,
                                                     len(val),
                                                     len(test)))
    return train_iter, val_iter, test_iter

def imdb(TEXT, LABEL, batch_size):
    """
    load IMDB (Large Moviue Review Dataset) dataset
    http://ai.stanford.edu/~amaas/data/sentiment/
    """
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train, max_size=20000)#, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=batch_size,
            shuffle=True, repeat=False)
    print('len(train):', len(train))
    print('len(test):', len(test))
    return train_iter, test_iter
