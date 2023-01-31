import collections
import json
import logging
import os
import time
import uuid
from pathlib import Path
from pprint import pprint as pp

import numpy as np
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
    language="multilingual", n_gram_range=(1, 2), min_topic_size=7, umap_model=UMAP(random_state=42), embedding_model=model
)
stopper = IStopper()


def search():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing phaze
    ic(data)
    query = data["text"].strip().lower()
    session["query"] = query
    uid = str(session["uid"])
    response = None
    try:
        response = store[uid].retrieve_top_k(processor(query), top_k=5)
    except:
        return jsonify({"docs": []})
    else:
        response = pipe.pipe_paint_docs(docs=response, query=query)
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
    """
    This part of the flow validates the file for corruption,

    performs the conversion to the .parquet

    sets the `email` | `text` | `date` columns
    """
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
    # TODO:
    # Serves as a proxy...
    # This will check the user's subscription as well as meta information
    # For now, it is dummy intermediate route that returns all the available route(s)
    return jsonify(
        [
            {"figure": "viewing_timeseries", "title": "TimeSeries", "premium": True},
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
    dfs = pipe.pipe_polar(
        dfr, txt_col_name=session["text"], fn=stopper, seps=[":", " "]
    )  # dataframe to which `IStopper` had been applied
    # Maybe there is a processed file already?
    # df = next(get_data(data_dir=cache_dir / uid, filename=fname.stem, ext=fname.suffix))
    # docs, times = df["text"].tolist(), df["datetime"].tolist()
    docs, times, raws = (
        [str(d) for d in list(dfs.select("text"))[0]],
        [str(d) for d in list(dfs.select("datetime"))[0]],
        [str(d) for d in list(dfr.select("text"))[0]],
    )
    # TODO: Move this wrapping functionality to the couchDB to make it async friendly
    engine = BM25L(processor=processor)
    engine.index([{"content": d, "timestamp": t, "raw": r} for d, t, r in zip(docs, times, raws)])
    store[uid] = engine
    # <--->

    # embeddings = model.encode(docs, show_progress_bar=True, device=devices[0])
    topics, probs = topmodel.fit_transform(docs, embeddings=None)  # we use model under the hood
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

    # TODO:
    session["plopics"] = topmodel.visualize_topics(width=1250, height=450)
    session["examples"] = topmodel.visualize_documents(raws, width=1250, height=450)
    session["tropics"] = topmodel.visualize_barchart(width=312.5, height=225, title="Topic Word Scores")

    # TODO:
    # Первый график всегда будет eager и ему тоже нужен будет response_type: docs
    # Остальные графики lazy после первого (первый тоже eager, т.к. клиент его должен сразу видеть)
    # Их
    response = [
        {
            "figure": json.loads(plotly.io.to_json(plotly_wordcloud(" ".join(docs), scale=1), pretty=True)),
            "lazy_figure_api": [
                {"api": "viewing_timeseries_examples", "title": "Representative documents per topic", "response_type": "docs"},
                {
                    "api": "viewing_timeseries_plopics",
                    "title": "Collinearity between topics",
                },
                {"api": "viewing_timeseries_tropics", "title": "Keyword ranking per topics"},
            ],
        },
        {
            "figure": json.loads(plotly.io.to_json(fig, pretty=True)),
        },
    ]
    return jsonify(response)


def view_timeseries_examples():
    return plotly.io.to_json(session["examples"], pretty=True)


def view_timeseries_tropics():
    return plotly.io.to_json(session["tropics"], pretty=True)


def view_timeseries_plopics():
    return plotly.io.to_json(session["plopics"], pretty=True)


def view_representation():
    uid = str(session["uid"])
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing phaze
    # {"api": "viewing_timeseries_examples", "topic_name": "..."}
    # TODO:
    query = data["topic_name"].strip().lower()
    api = data["api"].strip().lower()
    ic(api)
    if api != "default":  # TODO: default больше не будет
        return jsonify({"docs": []})
    engine = store[uid]
    response = None
    try:
        response = engine.retrieve_top_k(processor(query), top_k=100)
    except:
        return jsonify({"docs": []})

    response = pipe.pipe_paint_docs(docs=response, query=query)
    return jsonify({"docs": response})


def view_clustering():
    uid = str(session["uid"])
    filename = Path(session["filename"])
    # If the
    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    dfs = pipe.pipe_polar(dfr, txt_col_name=session["text"], fn=stopper, seps=[":", " "])
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
