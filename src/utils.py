import os
import numpy as np
import torch

from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager

#
def exists(val):
    return val is not None


def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load(
            "huggingface/pytorch-transformers", "tokenizer", "bert-base-cased"
        )
    return TOKENIZER


def get_bert():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load(
            "huggingface/pytorch-transformers", "model", "bert-base-cased"
        )
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL


# Pretrained Models


def tokenize(texts, model, add_special_tokens=True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts, add_special_tokens=add_special_tokens, padding=True, return_tensors="pt"
    )

    token_ids = encoding.input_ids
    return token_ids


def embed(*args):
    pass


def is_true_env_flag(env_flag):
    return os.getenv(env_flag, "false").lower() in ("true", "1", "t")


def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)


@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer
