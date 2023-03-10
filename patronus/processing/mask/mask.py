import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar


class IChunker(abc.ABC):
    def __init__(self, model: Callable, num_tokens: int = 384, window_size: int = 64):
        self.model = model
        self.num_tokens = num_tokens
        self.window_size = window_size

    @abc.abstractclassmethod
    def chunkify(self, **kwargs):
        pass


@dataclass
class IPath:
    runnsplitpath: ClassVar[str] = str(Path(__file__).parent.parent.parent / "recoiling" / "modelru.onnx")
    ennnsplitpath: ClassVar[str] = str(Path(__file__).parent.parent.parent / "recoiling" / "modelen.onnx")
    stopwordspath: ClassVar[str] = str(Path(__file__).parent.parent.parent / "recoiling" / "stopwords.txt")
    prefixwordspath: ClassVar[str] = str(Path(__file__).parent.parent.parent / "recoiling" / "prefixwords.txt")


__all__ = ["IChunker", "IPath"]
