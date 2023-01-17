from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from patronus.tooling import Email
from patronus.viewing.module import IFormat


def _process_polar(chunk: Dict, txt_col_name: str, fn: Callable):
    x = chunk[txt_col_name]
    y = fn(x)
    return y


def pipe_polar(df, txt_col_name, fn):
    df = df.select(
        [
            pl.struct(list(df.columns)).alias(txt_col_name).apply(partial(_process_polar, txt_col_name=txt_col_name, fn=fn)),
            pl.exclude(txt_col_name),
        ]
    )

    return df


def c_tf_idf(documents, m, ngram_range=(1, 1), stopwords: Iterable = None):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()  # num_docs x different_tokens
    w = t.sum(axis=1)  # (num_docs,) different tokens per document
    tf = np.divide(t.T, w)  #
    sum_t = t.sum(axis=0)  #
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)  #
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()  # list of words : words[pos] == "451" => 451 occurs @ pos
    labels = [str(d) for d in list(docs_per_topic.select(pl.col("Topic")))[0]]
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (
        df.groupby(["Topic"])
        .Doc.count()
        .reset_index()
        .rename({"Topic": "Topic", "Doc": "Size"}, axis="columns")
        .sort_values("Size", ascending=False)
    )
    return topic_sizes


def report_overall_topics(keywords: Dict, info: pd.DataFrame, filepath, plopics: Dict, topk: int = 5):
    name, count = [" | ".join(str(d).split("_")[1:]) for d in info["Name"]], list(info["Count"])
    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
    wb = writer.book
    ws = wb.add_worksheet("index")
    f = IFormat()
    TOP_COLS = [
        "Тематика",
        "Глобальная популярность тематики",
        "ТОП ключевых слов",
        "Локальная метрика (TF x IDF) в рамках тематики",
        "Локальный всплеск в рамках тематики",
        "Частота",
    ]

    for i, col in enumerate(TOP_COLS):
        ws.write(0, i, col, wb.add_format(f._headify))

    # --- TODO: Add coloring to the range cell(s)

    for topic_idx, top_words in keywords.items():
        pallete = next(f)

        topic_pos = int(topic_idx) + 1  # (e.g. [-1, 0, 1, 2] -> [0, 1, 2, 3])
        topic_name, topic_count = name[topic_pos], count[topic_pos]
        # Join the cell(s)
        # Assuming there are K keywords per topic.
        # TODO: Merge range is inclusive!
        start_idx = 1 + topic_pos * topk
        # TODO: Merge range on topics
        ws.merge_range(
            start_idx,
            0,
            start_idx + topk - 1,
            0,
            topic_name,
            wb.add_format(f.formify(pallete)),
        )
        # TODO: Merge range on topics
        ws.merge_range(
            start_idx,
            1,
            start_idx + topk - 1,
            1,
            topic_count,
            wb.add_format(f._centrify),
        )
        for i, item in enumerate(keywords[topic_idx]):
            word, score = item
            ws.write(start_idx + i, 2, word, wb.add_format(f.purify(pallete)))
            ws.write(start_idx + i, 3, score)

        # TODO: Add topic(s) for the spike

    writer.save()


def report_overall_snapshot(query: str, docs: List[Dict], filepath):
    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
    wb = writer.book
    ws = wb.add_worksheet("index")
    f = IFormat()
    ws.merge_range(0, 0, 0, 2, query, wb.add_format(f._centrify))
    TOP_COLS = ["Document", "Score", "Timestamp"]
    # skipping them here...
    for i, col in enumerate(TOP_COLS):
        ws.write(1, i, col, wb.add_format(f._headify))

    # Iterating over the snapshot
    for i, shot in enumerate(docs):
        pallete = next(f)
        snap, score, datetime = shot["text"], shot["upvote"], shot["timestamp"]
        ws.write(1 + i, 0, snap, wb.add_format(f.purify(pallete)))
        ws.write(1 + i, 1, score, wb.add_format(f._centrify))
        ws.write(1 + i, 2, datetime, wb.add_format(f._centrify))
    writer.save()


def send_over_email(
    attachments: Union[str, Path],
    author: str,
    receivers: Union[str, List[str]],
    subject,
    message: str,
):
    receivers = [receivers] if isinstance(receivers, str) else receivers
    mail = Email()
    mail.send(
        subject=subject,
        message=message,
        receivers=receivers,
        attachments=attachments,
        sender=author,
    )


__all__ = [
    "pipe_polar",
    "extract_top_n_words_per_topic",
    "extract_topic_sizes",
    "report_overall_topics",
    "report_overall_snapshot",
    "send_over_email",
]
