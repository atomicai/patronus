from typing import Callable, Optional

import numpy as np


def metrica_idf_recall(preds, labels, tokenizer: Optional[Callable] = None, mode: str = "hard"):
    pass


metrica = {"idf_recall": metrica_idf_recall}


__all__ = ["metrica"]
