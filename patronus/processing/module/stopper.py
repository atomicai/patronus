from pathlib import Path
from typing import List, Union

from patronus.processing.mask import IPath
from patronus.tooling import chunkify


class IStopper:
    def __init__(self, path: Union[Path, str] = None, do_lower_case: bool = True):
        self.store = set()
        self.do_lower_case = do_lower_case
        path = IPath.stopwordspath if path is None else path
        with open(str(path), "r") as fin:
            for word in chunkify(fin, sep="\n"):
                self.store.add(word.strip())

    def __call__(self, x, seps: List[str]):
        response = x.strip().lower() if self.do_lower_case else x.strip()
        for sep in seps:
            response = " ".join([w for w in response.split(sep) if w not in self.store])
        return response


__all__ = ["IStopper"]
