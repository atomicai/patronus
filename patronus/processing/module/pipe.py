"""
Here are all the auxiliary functions being used throughout the processing pipeline.
"""
from functools import partial
from typing import Dict

import polars as pl


def pipe_nullifier(x):
    return x == pl.Null() or str(x).strip() == ""


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


__all__ = ["pipe_nullifier", "pipe_polar"]
