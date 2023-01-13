import abc


class ILogger(abc.ABC):
    @abc.abstractmethod
    def log_metrics(self, d, **kwargs):
        pass


__all__ = ["ILogger"]
