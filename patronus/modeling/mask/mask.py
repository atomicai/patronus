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


class IKeyworder(abc.ABC):
    """Keyword extracting from the textual string"""

    @abc.abstractmethod
    def extract(self, x: Union[str, List[str]], **kwargs):
        pass


__all__ = ["IRI", "INI", "INNI", "ISFI", "IKeyworder"]
