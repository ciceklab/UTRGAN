import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import Sampler,BatchSampler,SubsetRandomSampler

class Basic_sampler(Sampler):
    def __init__(self, data):
        super().__init__(data)
        self.data= data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __iter__(self):
        return (i for i in range(self.data.shape[0]))
    
class sort_sampler(Sampler):
    def __init__(self, data, sort_key="utr_len"):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, seq_len) for i, seq_len in enumerate(data[sort_key].values)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)
    
class Bucket_Sampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.

    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self,
                 data,
                 batch_size,
                 drop_last=False,
                 sort_key='utr_len',
                 bucket_size_multiplier=100):
        self.data = data
        self.sampler = Basic_sampler(data)
        super().__init__(self.sampler, batch_size, drop_last)
        self.sort_key = sort_key
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(self.sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(self.sampler))
        self.bucket_sampler = BatchSampler(self.sampler, _bucket_size, False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = sort_sampler(self.data.iloc[bucket], self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return np.ceil(len(self.sampler) / self.batch_size)    
    

