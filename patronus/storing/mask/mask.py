import abc


class BaseDocStore(abc.ABC):
    def write(self, docs, index: str = None):
        pass


__all__ = ["BaseDocStore"]
