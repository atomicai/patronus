import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from patronus.etc import Document
from patronus.processing import KeywordProcessor
from patronus.tooling import Email, stl
from patronus.viewing.module import IFormat


def _process_polar(chunk: Dict, txt_col_name: str, fn: Callable, seps: List[str]):
    x = chunk[txt_col_name]
    y = fn(x, seps=seps)
    return y


def pipe_polar(df, txt_col_name, fn, seps):
    df = df.select(
        [
            pl.struct(list(df.columns))
            .alias(txt_col_name)
            .apply(partial(_process_polar, txt_col_name=txt_col_name, fn=fn, seps=seps)),
            pl.exclude(txt_col_name),
        ]
    )

    return df


def pipe_paint_docs(docs: List[Union[str, Document]], querix: List[str], prefix: List[str] = None):
    """
    This method performs two important function(s) for beautiful frontend rendering (with tabulation and newline(s) per each dialogue)
    1. Take the sequence of textual documents and find all occurences of the `query` returning them in the form of
    `{"lo": <offset_to_the_start>: "hi": <offset_to_the_end>}`
    2. As well as
    """
    response = [{"text": d.meta["raw"], "score": 0.5, "timestamp": d.meta["timestamp"]} for d in docs]
    kw = KeywordProcessor()
    # Given response "<prefix_1:> some text <prefix_2:> some other text ... <prefix_k:>" we would like to perform the following action(s):
    # 1. Before every <prefix_i:> insert the newline symbol "\n". This is required for beautiful rendering on the client side
    # 2. For every "<prefix_i:> - calculate offset in terms of number of chars up to beginning as well as ending
    # e.g. "hello world query:". For the "query:" prefix the response would be {"lo": 12, "hi": 18}
    if prefix is not None:
        for pref in prefix:
            kw.add_keyword(pref, "\n" + pref)

    preview = {k: i + 1 for i, k in enumerate(kw.get_all_keywords().values())}  # PREfix VIEW
    kw.add_keywords_from_list(querix)

    for doc in response:
        content = doc["text"].strip()
        posix = []
        chunk = []
        offset: int = 0
        occurence = kw.extract_keywords(content, span_info=True)
        prefcount = 0
        for step, hit in enumerate(occurence):
            w, lo, hi = hit  # w would be have been added newline separator in the case of prefix
            if w in preview:
                prefcount += 1

                posix.append({"lo": lo + prefcount, "hi": hi + prefcount, "color": preview[w]})
                if lo > 0:
                    sub = content[offset:lo]
                    chunk.append(sub + "\n")

                offset = lo
            else:
                posix.append({"lo": lo + prefcount, "hi": hi + prefcount})
        doc["highlight"] = posix
        doc["text"] = kw.replace_keywords(content)

    return response


def pipe_paint_kods(querix, engine):
    """
    Supposed to return the distribution of the word accross the whole corpus and highlight the `spike` over specific range
    """
    response = {}
    for q in querix:
        # Local "hike" is meant to be keyword
        docs = engine.retrieve_top_k(q, top_k=20)
        # TODO: rewrite scoring and add sorting argument(s)
        response[q] = [{"timestamp": d.meta["timestamp"], "value": random.randint(1, 11)} for d in docs]

    return response


def c_tf_idf(documents, m, ngram_range=(1, 1), stopwords: Iterable = None):
    """
    TODO: Make  iterable and propagate here
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()  # num_docs x different_tokens
    w = t.sum(axis=1)  # (num_docs,) different tokens per document
    tf = np.divide(t.T, w)  #
    sum_t = t.sum(axis=0)  #
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)  #
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    if sklearn.__version__.startswith("0."):
        words = count.get_feature_names()
    else:
        words = count.get_feature_names_out()
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

    offset = int(min(list(keywords.keys())))

    for topic_idx, top_words in keywords.items():
        pallete = next(f)

        topic_pos = int(topic_idx) + offset * (-1)  # Tricky way to add (+1) if (-1) is present and (+0) otherwise
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
    "pipe_paint_docs",
    "pipe_paint_kods",
    "extract_top_n_words_per_topic",
    "extract_topic_sizes",
    "report_overall_topics",
    "report_overall_snapshot",
    "send_over_email",
]
