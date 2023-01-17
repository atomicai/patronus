import abc
from typing import Dict, List, Union


class IRI(abc.ABC):
    """Information Retrieval Interface"""

    @abc.abstractmethod
    def retrieve_top_k(self, query: str, top_k: int = 10):
        pass


class INI(abc.ABC):
    """INdexing Interface"""

    @abc.abstractmethod
    def index(self, docs: List[Union[str, Dict[str, str]]]):
        pass


class INNI(INI):
    """Indexing using Neural Network Interface"""


class ISFI(INI):
    """Indexing using Speciall Features Interface"""


__all__ = ["IRI", "INI", "INNI", "ISFI"]
