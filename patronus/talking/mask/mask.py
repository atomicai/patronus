import abc


class IConsumer(abc.ABC):
    @abc.abstractmethod
    def consume(self, **kwargs):
        pass


class IProducer:
    @abc.abstractmethod
    def produce(self, msg, **kwargs):
        pass


class IExchange:
    pass


__all__ = ["IConsumer", "IProducer", "IExchange"]
