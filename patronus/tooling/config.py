import os
from pathlib import Path


class IConfig:
    def cache(self):
        res = os.environ.get("CACHE_DIR", "NONE")
        return Path(res)


__all__ = ["IConfig"]
