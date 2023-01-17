import abc


class IWorker(abc.ABC):
    @abc.abstractmethod
    def process(self, body, msg):
        pass


__all__ = ["IWorker"]
