from typing import List, Union

import yake

from patronus.modeling.mask import IKeyworder
from patronus.tooling import stl


class ISKeyworder(IKeyworder):
    def __init__(
        self,
        window_size: int = 2,
        stopwords=None,
        top_k: int = 22,
        dedupLim=0.22,
        dedupFunc="jaro_winkler",
        n: int = 1,
    ):
        super(ISKeyworder, self).__init__()
        stopwords = [] if stopwords is None else list(stl.NIterator(stopwords))

        self.model = yake.KeywordExtractor(
            n=n, dedupLim=0.22, dedupFunc="jaro_winkler", stopwords=stopwords, windowsSize=window_size, top=top_k
        )

    @classmethod
    def fire(cls, **config):
        return cls(**config)

    def extract(self, docs: Union[str, List[str]], **kwargs):
        if isinstance(docs, List):
            docs = " ".join(docs)
        response = self.model.extract_keywords(docs)
        return response


__all__ = ["ISKeyworder"]
