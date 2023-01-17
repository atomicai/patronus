from kombu import Exchange, Queue

exchange = Exchange("patronum", "direct")
inq = Queue("aiq", exchange=exchange, routing_key="inq")
ouq = Queue("air", exchange=exchange, routing_key="ouq")
task_queues = [inq, ouq]

__all__ = ["exchange", "inq", "ouq", "task_queues"]
