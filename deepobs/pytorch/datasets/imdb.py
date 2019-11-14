# -*- coding: utf-8 -*-
"""CIFAR-10 DeepOBS dataset."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext import data, datasets

from deepobs import config

from . import dataset


class imdb(dataset.DataSet):
    """DeepOBS data set class for the `IMDB' data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``50 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Methods:
      _make_dataloader: A helper that is shared by all three data loader methods.
  """

    def __init__(self,
                 batch_size,
                 train_eval_size=10000):
        """Creates a new IMDb dataset instance.
        Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size (``50 000`` for train, ``10 000``
            for test) the remainder is dropped in each epoch (after shuffling).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``10 000`` the size of the test set.
        """
        self._name = "imdb"
        self._train_eval_size = train_eval_size
        super(imdb, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length = 500)
        LABEL = data.Field(sequential=False)
    
        # make splits for data
        train, self.test_dataset = datasets.IMDB.splits(TEXT, LABEL, root=config.get_data_dir())
    
        train_dataset, valid_dataset = train.split(split_ratio=0.8)
    
        # build the vocabulary
        TEXT.build_vocab(train, max_size=10000)
        LABEL.build_vocab(train)
        
        train_loader, valid_loader, self.test_dataloader = self._make_all_dataloaders([train_dataset, valid_dataset, self.test_dataset])
        
        return self._make_dataloader(_ImdbHelper(train_loader), shuffle = True), self._make_dataloader(_ImdbHelper(valid_loader), shuffle=True)

    def _make_test_dataloader(self):
        return self._make_dataloader(_ImdbHelper(self.test_dataloader))
    
    def _make_all_dataloaders(self, datasets):

        iters = data.Iterator.splits(
            datasets, batch_size=self._batch_size)
        return (x for x in iters)


class _ImdbHelper(Dataset):
    def __init__(self, dataset_iter):
        texts = []
        lens = []
        labels = []
        for b in dataset_iter:
            texts.append(b.text[0])
            lens.append(b.text[1])
            labels.append(b.label - 1) # change class ids from 1,2 to 0,1
            
        max_len = max([t.size(1) for t in texts])
        texts_padded = []
        for t in texts:
            texts_padded.append(F.pad(t, pad=(0, max_len - t.size(1)), mode='constant', value=1))
            
        self.texts = torch.cat(texts_padded, dim=0)
        self.lens = torch.cat(lens, dim=0)
        self.labels = torch.cat(labels, dim = 0)
        
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return (self.texts[index], self.lens[index]), self.labels[index]


class BucketIteratorWrapper():
    def __init__(self, bucket_iterator):
        self.bucket_iterator = bucket_iterator
        self.dataset = bucket_iterator

    def __len__(self):
        return len(self.bucket_iterator)

    def __iter__(self):
        """
        To conform with the poutyne framework, we need to transform the
        output from BucketIterator.
        """
        for i, text_label_pair in enumerate(self.bucket_iterator):
            text = text_label_pair.text
            label = text_label_pair.label - 1  # change class labels from
            # [1,2] to [0,1]
            yield text, label
