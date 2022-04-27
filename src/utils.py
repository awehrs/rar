import os
from shutil import rmtree
from pathlib import Path
from typing import Callable, List, Tuple, Union

from einops import rearrange
import torch
from transformers import AutoModel, AutoTokenizer


# helper functions


def exists(val):
    return val is not None


# Downloading and using pytorch models.


def get_tokenizer(model: str) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def get_model(model: str) -> Callable:
    model = AutoModel.from_pretrained(model)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# Use pytorch models.


def tokenize(
    texts: Union[List, Tuple],
    *,
    tokenizer: Callable,
    max_length: int,
    add_special_tokens: bool = True,
) -> torch.tensor:

    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=add_special_tokens,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    token_ids = encoding.input_ids

    return token_ids


@torch.no_grad()
def embed(
    token_ids,
    encoder: Callable,
    return_cls_repr: bool = False,
    eps: float = 1e-8,
    pad_id: int = 0,
):

    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()

    outputs = encoder(
        input_ids=token_ids, attention_mask=mask, output_hidden_states=True
    )

    hidden_state = outputs.hidden_states[-1]

    # Return [CLS] as representation.
    if return_cls_repr:
        return hidden_state[:, 0]

    # Return mean over hidden state if no mask.
    if not exists(mask):
        return hidden_state.mean(dim=1)

    # Mean all tokens excluding [CLS], accounting for length.
    mask = mask[:, 1:]
    mask = rearrange(mask, "b n -> b n 1")

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)

    return masked_mean


def is_true_env_flag(env_flag):
    return os.getenv(env_flag, "false").lower() in ("true", "1", "t")


def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)
