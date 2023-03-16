"""
Here are all the auxiliary functions being used throughout the processing pipeline.
"""
from functools import partial
from typing import Dict

import dateparser as dp
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


def pipe_cmp(_df, date_column="datetime", pivot_date="2022-12-11 22:40:41", window_size: int = 1):
    pivot_day = dp.parse(pivot_date) if isinstance(pivot_date, str) else pivot_date
    start_day = pivot_day.day - window_size
    end_day = pivot_day.day + window_size
    start_date = dp.parse(
        f"{pivot_day.year}/{pivot_day.month}/{start_day} {pivot_day.hour}:{pivot_day.minute}:{pivot_day.second}"
    )
    end_date = dp.parse(f"{pivot_day.year}/{pivot_day.month}/{end_day} {pivot_day.hour}:{pivot_day.minute}:{pivot_day.second}")
    _df = _df.with_columns(
        [
            pl.when(pl.col(date_column).is_between(start=start_date, end=end_date, include_bounds=True))
            .then("Yes")
            .otherwise("No")
            .alias("match")
        ]
    )
    _df = _df.filter(pl.col("match") == "Yes")
    return _df.drop(["match"])


__all__ = ["pipe_nullifier", "pipe_polar"]
