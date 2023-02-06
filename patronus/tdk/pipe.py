from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from patronus.etc import Document
from patronus.tooling import Email
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
    Take the sequence of textual documents and find all occurences of the `query` returning them in the form of
    `{"lo": <offset_to_the_beginning>: "hi": <offset_to_the_end>}`
    """
    response = [{"text": d.meta["raw"], "score": 0.5, "timestamp": d.meta["timestamp"]} for d in docs]

    for doc in response:
        posix = []
        content = doc["text"].lower()
        offset = -1
        pre, pos = -1, 0
        for _, pref in enumerate(prefix):
            pos = content.find(pref, pos)
            if pre > -1:
                if pos > -1:
                    paragraph = doc[pre + offset : pos + offset]
                else:
                    paragraph = doc[pre + offset :]
                doc = doc.replace(paragraph, paragraph + "\n")
            if pos > -1:
                posix.append({"lo": pre + offset, "hi": pos + offset})
                pre = pos
                pos += len(pref)
                offset += 1

        for query in querix:
            pos, l = 0, len(query)
            while pos != -1:
                pos = content.find(query, pos)
                if pos != -1:
                    posix.append({"lo": pos, "hi": pos + l - 1, "color": "color1"})
                    pos += l

        doc["highlight"] = posix
    return response


# len(highlight) * 2 <u></u>


# def pipe_paint_docs(docs: List[Union[str, Document]], prefix=None, querix=None):
#     docs = [
#         {"text": d.meta["raw"], "score": 0.5, "timestamp": d.meta["timestamp"]} if isinstance(d, Document) else {"text": d}
#         for d in docs
#     ]
#     for document in docs:
#         doc = document["text"]
#         posix = []
#         pre, pos = 0, 0
#         for pre in prefix:
#             pos = doc.find(pre, pos)
#             sub = doc[pre:pos] if pos >= 0 else doc
#             # TODO: Perform search here
#             for query in querix:
#                 _pos = 0
#                 while _pos != -1:
#                     _pos = sub.find(query, _pos)
#                     dx = 0 if pos <= 0 else pos
#                     if _pos != -1:
#                         posix.append({"lo": _pos + dx, "hi": _pos + len(query) + dx, "color": "color2"})
#                         _pos += len(query)
#             if pos != -1:
#                 posix.append(pos)
#                 pre = pos
#                 pos += len(pre)
#         posix = sorted(posix)
#         for offset, (lo, hi) in enumerate(zip(posix[:-1], posix[1:])):
#             paragraph = doc[lo + offset : hi + offset]
#             doc = doc.replace(paragraph, paragraph + "\n")
#             # TODO: We can find all the queries here
#             # The queries would be highlighted and formatted without needing to remember offset later

#             posix.append({"lo": lo + offset, "hi": hi + offset, "color": "color1"})

#         document["text"] = doc
#         document["highlight"] = light

#     return docs


def c_tf_idf(documents, m, ngram_range=(1, 1), stopwords: Iterable = None):
    """
    TODO: Make IStopper iterable and propagate here
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
    "pipe_prefix_docs",
    "extract_top_n_words_per_topic",
    "extract_topic_sizes",
    "report_overall_topics",
    "report_overall_snapshot",
    "send_over_email",
]
