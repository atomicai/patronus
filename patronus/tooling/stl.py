class NIterator(object):
    __slots__ = ("_is_next", "_the_next", "it")

    def __init__(self, it):
        self.it = iter(it)
        self._is_next = None
        self._the_next = None

    def has_next(self) -> bool:
        if self._is_next is None:
            try:
                self._the_next = next(self.it)
            except:
                self._is_next = False
            else:
                self._is_next = True
        return self._is_next

    def next(self):
        if self._is_next:
            response = self._the_next
        else:
            response = next(self.it)
        self._is_next = None
        return response


__all__ = ["NIterator"]
