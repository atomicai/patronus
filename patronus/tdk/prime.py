import collections
import json
import logging
import os
import time
import uuid
from pathlib import Path
from pprint import pprint as pp

import plotly
import polars as pl
import pyarrow.parquet as pq
import random_name
import torch
from bertopic import BERTopic
from flask import jsonify, render_template, request, send_file, session
from icecream import ic
from kombu import Connection, Consumer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from werkzeug.utils import secure_filename

from patronus.modeling.module import BM25L
from patronus.processing import IStopper
from patronus.storing.module import SQLDocStore
from patronus.tdk import pipe
from patronus.tooling import get_data, initialize_device_settings
from patronus.viewing.module import plotly_wordcloud

logger = logging.getLogger(__name__)

cache_dir = Path(os.getcwd()) / ".cache"


devices, n_gpu = initialize_device_settings(use_cuda=torch.cuda.is_available(), multi_gpu=False)


processor = lambda x: x.lower().strip().split(" ")

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=devices[0])
store = collections.defaultdict()
topmodel = BERTopic(
    language="multilingual",
    n_gram_range=(1, 2),
    min_topic_size=7,
    umap_model=UMAP(random_state=42),
)
stopper = IStopper()


def search():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing phaze
    ic(data)
    query = data["text"]
    session["query"] = query
    uid = str(session["uid"])
    response = None
    try:
        response = store[uid].retrieve_top_k(processor(query), top_k=5)
    except:
        return jsonify({"docs": []})
    else:
        response = [{"text": d.meta["raw"], "score": 0.5, "timestamp": d.meta["timestamp"]} for d in response]
    # highlight: [{"lo": 0, "hi": 5}, {"lo": 8, "hi": 13}, {"lo": 124, "hi": 1241}]
    return jsonify({"docs": response})


def upload():
    response = {}
    logger.info("welcome to upload`")
    xf = request.files["file"]
    filename = secure_filename(xf.filename)
    ic(f"{filename}")
    if "uid" not in session.keys():
        uid = uuid.uuid4()
        session["uid"] = str(uid)
    else:
        uid = session["uid"]
    if not Path(filename).suffix in (".csv", ".xlsx", ".txt"):
        response["is_suffix_ok"] = False
    else:
        if not (cache_dir / str(uid)).exists():
            (cache_dir / str(uid)).mkdir(parents=True, exist_ok=True)
        destination = cache_dir / str(uid) / filename
        response["is_suffix_ok"] = True
        response["is_file_corrupted"] = False
        if Path(filename).suffix in (".csv", ".xlsx"):
            xf.save(str(destination))
        session["filename"] = filename
        response["filename"] = filename

    # --- persist the state ---

    return response


def iload():
    data = request.get_json()
    pp("ILOADing ... \n")
    pp(data)
    #
    uid = session["uid"]
    filename = Path(session["filename"])
    df = next(
        get_data(
            data_dir=cache_dir / uid,
            filename=filename.stem,
            ext=filename.suffix,
            engine="polars",
        )
    )
    # df = pipe.pipe_polar(df, txt_col_name=data["text"], fn=stopper)
    arr = df.to_arrow()
    try:
        pq.write_table(arr, cache_dir / uid / f"{filename.stem}.parquet")
        # text_column_name, datetime_column_name, email = data["text"], data["datetime"], data["email"]
        session["email"] = data.get("email", None)
        session["text"] = data.get("text", "text")
        session["datetime"] = data.get("datetime", "datetime")
    except:
        return jsonify({"is_date_column_ok": False, "is_text_column_ok": False})

    # NEED to do something with preprocessed DB
    # TODO:
    #  (1) Предобработанный файл сохраняем и добавляем в кэш
    #  (2)
    return jsonify({"is_date_column_ok": True, "is_text_column_ok": True})


def download(filename):
    ic(session["uid"])
    uid = str(session["uid"])
    fname = Path(Path(filename).stem + ".xlsx")
    destination = cache_dir / str(uid) / str(fname)
    return send_file(str(destination), as_attachment=True)


def view():
    return jsonify(
        [
            {"figure": "viewing_timeseries", "title": "TimeSeries", "premium": True},
            {"figure": "viewing_clustering", "title": "Clustering", "premium": True},
        ]
    )


def view_timeseries():
    uid = str(session["uid"])
    filename = Path(session["filename"])
    # fname = Path(filename)
    # TODO: Push here to the queue
    # Here we fetch from queue (BROKER) until the required message is receieved

    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")  # raw without cleaning, etc..
    # <--->
    dfs = pipe.pipe_polar(dfr, txt_col_name=session["text"], fn=stopper)  # dataframe to which `IStopper` had been applied
    # Maybe there is a processed file already?
    # df = next(get_data(data_dir=cache_dir / uid, filename=fname.stem, ext=fname.suffix))
    # docs, times = df["text"].tolist(), df["datetime"].tolist()
    docs, times, raws = (
        [str(d) for d in list(dfs.select("text"))[0]],
        [str(d) for d in list(dfs.select("datetime"))[0]],
        [str(d) for d in list(dfr.select("text"))[0]],
    )
    engine = BM25L(processor=processor)
    engine.index([{"content": d, "timestamp": t, "raw": r} for d, t, r in zip(docs, times, raws)])
    store[uid] = engine
    # <--->

    embeddings = model.encode(docs, show_progress_bar=True, device=devices[0])
    topics, probs = topmodel.fit_transform(docs, embeddings=embeddings)
    # Here is the ideal place to visualize count and save the file to return it to client
    info = topmodel.get_topic_info()
    docs_per_topic = (
        pl.DataFrame({"Doc": docs, "Topic": topics, "Id": range(len(docs))})
        .groupby("Topic")
        .agg(pl.col("Doc").apply(lambda x: " ".join(x)))
    )
    tf_idf, count = pipe.c_tf_idf([str(d) for d in list(docs_per_topic.select(pl.col("Doc")))[0]], m=len(docs))
    top_n_words = pipe.extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=5)

    topics_over_time = topmodel.topics_over_time(docs, times, nr_bins=20)
    plopics = pl.from_pandas(topics_over_time)
    plopics = (
        plopics.sort("Frequency")
        .groupby("Topic")
        .agg(
            [
                pl.col("Words").last(),
                pl.col("Frequency").last(),
                pl.col("Timestamp").last(),
            ]
        )
    ).to_dict()

    plopics = {k: list(v) for k, v in plopics.items()}

    # TODO: use timestamp from topics_over_time!
    report_filepath = Path(os.getcwd()) / ".cache" / uid / str(filename.stem + ".xlsx")
    pipe.report_overall_topics(
        keywords=top_n_words,
        info=info,
        filepath=str(report_filepath),
        plopics=plopics,
        topk=5,
    )
    # TODO: send the report over email to the recievers with attachment(s) = [report_filepath]
    if session.get("email", None) is not None:
        pipe.send_over_email(
            attachments=[report_filepath],
            receivers=[str(session["email"])],
            subject="Анализ обращений по ключевым словам в разрезе разных тематик",
            message="Во вложении файл с аналитикой.\nС уважением, Команда корневых причин.",
            author="itarlinskiy@yandex.ru",
        )
    fig = topmodel.visualize_topics_over_time(topics_over_time, top_n_topics=10)

    response = [
        {
            "figure": json.loads(plotly.io.to_json(fig, pretty=True)),
            "lazy_figure_api": [
                {"api": "viewing_timeseries_examples", "title": "Representative documents per topic"},
                {"api": "viewing_timeseries_plopics", "title": "Collinearity between topics"},
            ],
        },
        {
            "figure": json.loads(plotly.io.to_json(fig, pretty=True)),
            "keywords": [
                {
                    "data": [f"{str(i)}", f"{str(i + 1)}", f"{str(i + 2)}"],
                    "title": f"title_{str(i)}",
                    "api": "viewing_keywording",
                }
                for i in range(15)
            ],
        },
    ]
    return jsonify(response)
    # return plotly.io.to_json(fig, pretty=True)


def view_timeseries_examples():
    time.sleep(4)
    uid = str(session["uid"])
    filename = Path(session["filename"])
    # If the
    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")

    dfs = pipe.pipe_polar(dfr, txt_col_name=session["text"], fn=stopper)
    flow = list(dfs.select("text"))[0]
    text = " ".join(flow)
    fig = plotly_wordcloud(text, scale=1)

    return plotly.io.to_json(fig, pretty=True)


def view_timeseries_plopics():
    uid = str(session["uid"])
    filename = Path(session["filename"])
    # If the
    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    dfs = pipe.pipe_polar(dfr, txt_col_name=session["text"], fn=stopper)
    flow = list(dfs.select("text"))[0]
    text = " ".join(flow)
    fig = plotly_wordcloud(text, scale=1)
    return plotly.io.to_json(fig, pretty=True)


def view_clustering():
    uid = str(session["uid"])
    filename = Path(session["filename"])
    # If the
    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    dfs = pipe.pipe_polar(dfr, txt_col_name=session["text"], fn=stopper)
    flow = list(dfs.select("text"))[0]
    text = " ".join(flow)
    fig = plotly_wordcloud(text, scale=1)
    return plotly.io.to_json(fig, pretty=True)


def snapshot():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing phase
    ic(data)
    # retrieve filename
    filename = Path("snapshot" + "_" + str(session["filename"]))
    filepath = cache_dir / str(session["uid"]) / (str(filename.stem) + ".xlsx")
    pipe.report_overall_snapshot(query=session["query"], docs=data["docs"], filepath=filepath)
    return jsonify({"filename": str(filename)})
    # Supposed to return the json with the filename to download


__all__ = [
    "upload",
    "iload",
    "view_timeseries",
    "view_clustering",
    "snapshot",
    "search",
]
