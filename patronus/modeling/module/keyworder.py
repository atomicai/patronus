from pathlib import Path
from typing import List, Union

from keybert import KeyBERT

from patronus.modeling.mask import IKeyworder
from patronus.tooling import stl


class ISKeyworder(IKeyworder):
    def __init__(self, model_path: Union[str, Path], use_mmr: bool = False, diversity=0.7, stopwords=None):
        super(ISKeyworder, self).__init__()
        self.model = KeyBERT(str(Path(model_path)))
        self.config = {"use_mmr": use_mmr, "diversity": diversity}
        self.stopwords = [] if stopwords is None else list(stl.NIterator(stopwords))

    def extract(self, docs: Union[str, List[str]], **kwargs):
        response = self.model.extract_keywords(docs, stop_words=self.stopwords, **self.config)
        return response


__all__ = ["ISKeyworder"]
