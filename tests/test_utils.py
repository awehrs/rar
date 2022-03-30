from src.utils import embed, tokenize

import torch
import torch.nn.functional as F

MODEL = "bert-base-uncased"
SEQ_LEN = 200


def test_embed():
    """
    Test that tokenizing/embedding produce different
    outputs for CLS versus non-CLS representations.
    """

    sentence1 = "This is the first sentence."
    sentence2 = "This is the second sentence with a few extra words."

    ids = tokenize(texts=[sentence1, sentence2], model=MODEL)

    text_len = ids.shape[-1]
    padding = SEQ_LEN - text_len
    ids = F.pad(ids, (0, padding))

    cls_embeddings = embed(model=MODEL, token_ids=ids, return_cls_repr=True)
    mean_embeddings = embed(model=MODEL, token_ids=ids, return_cls_repr=False)

    assert (cls_embeddings.size() == mean_embeddings.size()) and (
        not torch.equal(cls_embeddings, mean_embeddings)
    )
