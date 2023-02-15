from copy import copy
from itertools import cycle
from typing import List

from patronus.tooling import stl


class IMap(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(IMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(IMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(IMap, self).__delitem__(key)
        del self.__dict__[key]


class IFormat(IMap):
    pallete = [
        "#54bebe",
        "#76c8c8",
        "#98d1d1",
        "#badbdb",
        "#dedad2",
        "#e4bcad",
        "#df979e",
        "#d7658b",
        "#c80064",
    ]

    _headify = {
        "bold": True,
        "text_wrap": True,
        "valign": "vcenter",
        "align": "center",
        "fg_color": "#D7E4BC",
        "border": 1,
    }

    _formify = {
        "fg_color": "#99badd",
        "font_name": "Lucida Grande",
        "text_wrap": True,
        "font_color": "#001440",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    }

    _purify = {
        "align": "center",
        "valign": "vcenter",
        "fg_color": "#D7E4BC",
        "font_name": "Lucida Grande",
        "border": 1,
    }

    _centrify = {
        "align": "center",
        "valign": "vcenter",
        "font_name": "Lucida Grande",
        "border": 1,
    }

    _signify = {
        "align": "center",
        "valign": "vcenter",
        "font_name": "Lucida Grande",
        "border": 1,
        "bold": True,
    }

    _italify = {
        "align": "center",
        "valign": "vcenter",
        "font_name": "Lucida Grande",
        "bord": 1,
        "italic": True,
    }

    def __init__(self, pallete: List[str] = None):
        if pallete is None:
            pallete = self.pallete
        self.it = stl.NIterator(cycle(pallete))

    def formify(self, pallete):
        pal = copy(self._formify)
        pal["fg_color"] = pallete
        return pal

    def purify(self, pallete):
        pal = copy(self._purify)
        pal["fg_color"] = pallete
        return pal

    def __iter__(self):
        return self.it

    def __next__(self):
        return self.it.next()


__all__ = ["IFormat"]
