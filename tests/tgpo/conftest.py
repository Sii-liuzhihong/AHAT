"""Shared local tokenizer fixture for executable tensor and training-loop checks."""
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    vocab = {'[PAD]': 0, '[UNK]': 1, **{f'[unused{i}]': i for i in range(2, 32)},
             **{chr(i): i for i in range(32, 127)}, '\n': 127}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', padding_side='left')

