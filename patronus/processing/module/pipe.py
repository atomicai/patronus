"""
Here are all the auxiliary functions being used throughout the processing pipeline.
"""
import string
from functools import partial
from typing import ClassVar, Dict, Optional

import dateparser as dp
import polars as pl

from patronus.etc.schema import Document
from patronus.tooling import stl


def pipe_cmp_date(dx: ClassVar[Document], dy: ClassVar[Document]):
    """
    Comparator to sort the given `documents` by datetime attribute, located in `meta` subfield
    Args:
        dx (_type_): Document
        dy (_type_): Document
    """
    predate = dx.meta["timestamp"]
    curdate = dy.meta["timestamp"]
    if predate >= curdate:
        return 1
    return -1


def pipe_std_parse(d):
    time = None
    if isinstance(d, Document):
        time = d.meta["timestamp"]
    else:
        time = str(d)
    _dt = dp.parse(time)
    d, m, y = _dt.strftime("%d"), _dt.strftime("%m"), _dt.strftime("%y")
    h, m, s = _dt.strftime("%H"), _dt.strftime("%M"), _dt.strftime("%S")
    return f"{d}/{m}/{y} {h}:{m}:{s}"


def pipe_nullifier(x):
    return x == pl.Null() or str(x).strip() == ""


def std_date(x):
    _dt = dp.parse(x, settings={'DATE_ORDER': 'MDY'})
    d, m, y = _dt.strftime("%d"), _dt.strftime("%m"), _dt.strftime("%Y")
    hh, mm, ss = _dt.strftime("%H"), _dt.strftime("%M"), _dt.strftime("%S")
    return f"{d}/{m}/{y} {hh}:{mm}:{ss}"


def std_map(mapping: Dict):
    cursor = pl.element()
    for pre, nex in mapping.items():
        cursor = cursor.str.replace_all(pre, str(nex), literal=True)
    return cursor


# TODO: add normal form checking
def std_replace(x: str, wordlist):
    r = []
    for w in x.split(" "):
        w = w.strip()
        _w = "".join([ch for ch in w if ch not in string.punctuation])
        if _w.strip().lower() not in wordlist and not _w.strip().isdigit():
            r.append(w)
    return " ".join(r)


def std_cast(x: str):
    return dp.parse(x)


def pipe_silo(df, txt_col_name, syms, wordlist, date_col_name: str = None):  # TODO: perform apply to another column as well
    lidx: int = None
    for i, sep in enumerate(syms):
        if sep != " ":
            pre_col_name = f"re{str(lidx)}" if lidx is not None else txt_col_name
            df = df.with_column(pl.col(pre_col_name).str.split(sep).alias(f"_re{str(i)}"))
            df = df.with_column(pl.col(f"_re{str(i)}").arr.join(" ").alias(f"re{str(i)}"))
            df = df.drop([f"_re{str(i)}"])
            if pre_col_name != txt_col_name:
                df = df.drop([pre_col_name])
            lidx = i
        df = df.with_column(pl.col(f"re{str(lidx)}").apply(partial(std_replace, wordlist=set(wordlist))).alias("silo"))
    if date_col_name is not None:
        df = df.with_column(pl.col(date_col_name).apply(std_cast).alias(f"redate")).with_column(
            pl.col("redate").cast(pl.Datetime)
        )

    return df


def _silo(df, txt_col_name, wordlist):
    df = df.with_column(pl.col(txt_col_name).apply(partial(std_replace, wordlist=set(wordlist))).alias("silo"))
    return df


def pipe_polar(df, txt_col_name, fn, seps):
    """
    The wrapper below is responsible for taking any callable fun and applying it to the given column.
    """

    def process(chunk: Dict, column_name, fun, separators):
        sub = chunk[column_name]
        response = fun(sub, seps=separators)
        return response

    df = df.select(
        [
            pl.struct(list(df.columns))
            .alias(txt_col_name)
            .apply(partial(process, column_name=txt_col_name, fun=fn, separators=seps)),
            pl.exclude(txt_col_name),
        ]
    )

    return df


def pipe_bound(
    _df,
    date_column="datetime",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_id: Optional[int] = None,
    topic_column="topic",
):
    start = dp.parse(start_date) if isinstance(start_date, str) else start_date
    end = dp.parse(end_date) if isinstance(end_date, str) else end_date
    _df = _df.with_columns(
        [pl.when(pl.col(date_column).is_between(start=start, end=end, closed="both")).then("Yes").otherwise("No").alias("match")]
    )
    if topic_id is None:
        _df = _df.filter(pl.col("match") == "Yes")
    else:
        _df = _df.filter(pl.col("match") == "yes" & pl.col(topic_column) == topic_id)
    return _df.drop(["match"])


__all__ = ["pipe_nullifier", "pipe_polar", "pipe_bound", "pipe_cmp_date", "std_map", "pipe_std_parse", "std_date"]
