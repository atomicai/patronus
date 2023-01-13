import abc


class IEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, **kwargs):
        pass
