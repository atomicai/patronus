import ast
from pathlib import Path
from typing import List, Union

import numpy as np
import random_name
import simplejson


def chunkify(f, chunksize=10_000_000, sep="\n"):
    """
    Read a file separating its content lazily.

    Usage:

    >>> with open('INPUT.TXT') as f:
    >>>     for item in chunkify(f):
    >>>         process(item)
    """
    chunk = None
    remainder = None  # data from the previous chunk.
    while chunk != "":
        chunk = f.read(chunksize)
        if remainder:
            piece = remainder + chunk
        else:
            piece = chunk
        pos = None
        while pos is None or pos >= 0:
            pos = piece.find(sep)
            if pos >= 0:
                if pos > 0:
                    yield piece[:pos]
                piece = piece[pos + 1 :]
                remainder = None
            else:
                remainder = piece
    if remainder:  # This statement will be executed iff @remainder != ''
        yield remainder


def get_data(
    data_dir: Union[Path, str],
    filename: str,
    embedding_field="embedding",
    load_embedding=True,
    ext=".json",
    parse_meta: bool = False,
    lazy: bool = False,
    sep: str = ",",
    encoding: str = "utf-8-sig",
    as_record: bool = False,
    rename_columns: dict = None,
    engine: str = "pandas",
    **kwargs,
):
    assert engine in ("pandas", "polars")
    if engine == "polars":
        import polars as pd
    else:
        import pandas as pd
    data_dir = Path(data_dir)
    db_filename = filename
    db_filepath = data_dir / (db_filename + ext)

    if ext in (".csv", ".tsv", ".xlsx", ".pickle", ".gz"):
        columns_needed = list(rename_columns.keys()) if rename_columns else None
        if ext == ".xlsx":
            df = pd.read_excel(db_filepath, engine="openpyxl") if engine == "pandas" else pd.read_excel(db_filepath)
        elif ext in (".tsv", ".csv"):
            if engine == "pandas":
                df = pd.read_csv(
                    db_filepath,
                    encoding=encoding,
                    usecols=columns_needed,
                    skipinitialspace=True,
                    sep=sep,
                    **kwargs,
                )
            else:
                df = pd.read_csv(db_filepath, encoding=encoding, sep=sep)
                df = df[columns_needed] if columns_needed else df
        elif ext in (".pickle"):
            df = pd.read_pickle(db_filepath, **kwargs)
        else:
            df = pd.read_csv(db_filepath, header=0, error_bad_lines=False, **kwargs)
        if rename_columns is not None:
            df = df.rename(rename_columns) if rename_columns else df
        if as_record:
            yield df.to_dict(orient="records")
        else:
            yield df
        raise StopIteration()
    with open(str(db_filepath), "r", encoding=encoding) as j_ptr:
        if lazy:
            for jline in j_ptr:
                yield simplejson.loads(jline)
        else:
            docs = simplejson.load(j_ptr)

    if lazy:
        raise StopIteration()

    if parse_meta:
        for d in docs:
            d["meta"] = ast.literal_eval(d["meta"])

    if embedding_field is not None:
        if load_embedding:
            index_filename = filename + "_index" + ".npy"
            index_filepath = data_dir / index_filename
            embeddings = np.load(str(index_filepath))
            for iDoc, iEmb in zip(docs, embeddings):
                iDoc[embedding_field] = iEmb
        else:
            for iDoc in docs:
                iDoc[embedding_field] = np.nan

    yield docs


def save_data(
    data,
    data_dir: Union[str, Path],
    filename: str = None,
    embedding_field_or_cols: List[str] = None,
    db_field_or_cols: List[str] = None,
    ext=".json",
    engine: str = "pandas",
):
    assert engine in ("pandas", "polars")
    if engine == "polars":
        import polars as pd
    else:
        import pandas as pd
    data_dir = Path(data_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_filename = random_name.generate_name() if filename is None else str(filename)

    db_filepath = data_dir / (db_filename + ext)

    assert ext in (
        ".json",
        ".csv",
    ), f'Extension "{ext}" is not supported yet. Use ".json" or ".csv"'
    if ext == ".json":
        with open(str(db_filepath), "w", encoding="utf-8-sig") as j_ptr:
            simplejson.dump(data, j_ptr, indent=4, ensure_ascii=False, ignore_nan=True)
    elif ext == ".csv":
        pd.DataFrame(data, columns=db_field_or_cols).to_csv(db_filepath, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f'Extension {ext} is not yet supported: Pick one of ".json", ".csv", ".xlsx"')


__all__ = ["get_data", "save_data", "chunkify"]
