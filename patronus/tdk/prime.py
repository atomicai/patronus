import collections
import json
import logging
import os
import uuid
from pathlib import Path

import plotly
import polars as pl
import pyarrow.parquet as pq
import random_name
import torch
from bertopic import BERTopic
from flask import jsonify, request, send_file, session
from icecream import ic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP
from werkzeug.utils import secure_filename

from patronus.modeling.module import BM25Okapi, ISKeyworder
from patronus.processing import IPrefixer, IStopper
from patronus.processing import pipe as ppipe
from patronus.tdk import pipe
from patronus.tooling import get_data, initialize_device_settings
from patronus.viewing.module import plotly_wordcloud

logger = logging.getLogger(__name__)

cache_dir = Path(os.getcwd()) / ".cache"


devices, n_gpu = initialize_device_settings(use_cuda=torch.cuda.is_available(), multi_gpu=False)


def processor(x, seps=("_", " ")):
    x = x.lower().strip()
    for sep in seps:
        if sep != " ":
            x = " ".join(x.split(sep))

    return x.split(" ")


class OS:
    def __init__(self, klass, config):
        self.klass = klass
        self.config = config

    def fire(self, **kwargs):
        for k, v in kwargs.items():
            self.config[k] = v
        return self.klass(**self.config)


model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=devices[0])
store = collections.defaultdict()
botpic = OS(
    BERTopic,
    config=dict(
        umap_model=UMAP(
            n_neighbors=22,
            random_state=42,
            min_dist=0.0,
            n_components=5,
            init="random",
            metric="cosine",
        ),
        n_gram_range=(1, 2),
        language="multilingual",
    ),
)
stopper = IStopper()
prefixer = IPrefixer()
iworder = ISKeyworder(stopwords=stopper)


def search():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    # Placeholder to retrieve all the document(s) from the indexing stage
    ic(data)
    query = data["text"].strip().lower()
    session["query"] = query
    querix = processor(query)
    uid = str(session["uid"])
    response = None
    try:
        response = store[uid].retrieve_top_k(processor(query), top_k=5)
    except:
        return jsonify({"docs": []})
    else:
        response = pipe.pipe_paint_docs(docs=response, querix=querix, prefix=prefixer)
    return jsonify({"docs": response})


def upload():
    response = {}
    logger.info("welcome to upload`")
    xf = request.files["file"]
    prefixname = random_name.generate_name()
    filename = secure_filename(prefixname + xf.filename)
    ic(f"{filename}")
    if "uid" not in session.keys():
        uid = uuid.uuid4()
        session["uid"] = str(uid)
    else:
        uid = session["uid"]

    if not (cache_dir / str(uid)).exists():
        (cache_dir / str(uid)).mkdir(parents=True, exist_ok=True)
    destination = cache_dir / str(uid) / filename
    fname, fpath = Path(filename), Path(destination)
    df, columns = None, None
    is_suffix_ok, is_file_corrupted = True, False
    if fpath.suffix not in (".xlsx", ".csv"):
        is_suffix_ok = False
    else:
        try:
            xf.save(str(destination))
            df = next(
                get_data(
                    data_dir=cache_dir / uid,
                    filename=fname.stem,
                    ext=fname.suffix,
                    engine="polars",
                )
            )
        except:
            is_file_corrupted = True
        else:
            is_file_corrupted = False

    if is_suffix_ok and not is_file_corrupted:  # is suffix_ok is also false
        columns = [str(_) for _ in list(df.columns)]
        arr = df.to_arrow()
        pq.write_table(arr, cache_dir / uid / f"{fname.stem}.parquet")
        session["filename"] = filename
        response["filename"] = filename
        response["text_columns"] = columns
        response["datetime_columns"] = columns
    elif is_suffix_ok:
        msg = "There are some technical issues processing file. Please make sure the file is not corrupted"
        response["error"] = msg
        ic(msg)
    else:
        msg = f"The file format {fname.suffix} is not yet supported. The supported file formats are \".csv\" and \".xlsx\""
        response["error"] = msg
        ic(msg)
    response["is_suffix_ok"] = is_suffix_ok
    response["is_file_corrupted"] = is_file_corrupted
    return response


def iload():
    """
    This route will scan the file and perform the "lazy" way to push it to the database

    sets the `email` | `text` | `date` columns
    """
    data = request.get_json()
    #
    uid = session["uid"]
    filename = Path(session["filename"])
    text_column, datetime_column, num_clusters = (
        data.get("text", None),
        data.get("datetime", None),
        data.get("num_clusters", None),
    )
    is_text_ok, is_date_ok, is_num_clusters_ok = False, False, False
    df = pl.scan_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    if text_column is not None and text_column in df.columns:
        session["text"] = text_column
        is_text_ok = True
    if datetime_column is not None and datetime_column in df.columns:
        session["datetime"] = datetime_column
        is_date_ok = True

    if num_clusters is not None and int(num_clusters) > 0 and int(num_clusters) < 30:
        session["num_clusters"] = int(num_clusters)
        is_num_clusters_ok = True
    else:
        logger.info(
            f"The number of clusters specified {str(num_clusters)} is not in a reasonable range. Setting default to {str(7)}"
        )
        session["num_clusters"] = 7

    session["email"] = data.get("email", None)

    if is_date_ok and is_text_ok:
        if is_num_clusters_ok:
            return jsonify({"success": "In progress to the moon 🚀"})
        else:
            return jsonify({"success": "In progress to the moon 🚀 with d"})
    else:
        return jsonify({"Error": "Back to earth ♁. Fix the column name(s) 🔨"})

    return jsonify({"is_date_column_ok": is_date_ok, "is_text_column_ok": is_text_ok})


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
        ]
    )


def view_timeseries():
    uid = str(session["uid"])
    filename = Path(session["filename"])
    df = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")  # raw without cleaning, etc..
    tecol, dacol = session["text"], session["datetime"]
    df = df.filter(~pl.col(tecol).is_null() & ~pl.col(dacol).is_null())
    df = ppipe.pipe_silo(df, tecol, syms=[":"], wordlist=set(stopper), date_col_name=dacol)
    print("silo")
    df = (
        df.with_row_count()
        .with_columns([pl.col("row_nr").last().over("silo").alias("idx_per_unique")])
        .filter(pl.col("row_nr") == pl.col("idx_per_unique"))
    )
    if df.shape[0] >= int(os.environ.get("TOP_N_ROWS", 10_000)):
        _warning_volume = df.shape[0]
        df = df.sample(int(os.environ.get("TOP_N_ROWS", 10_000)))
        _clipped_volume = df.shape[0]
        logger.info(
            f"The overall number of appeals {_warning_volume} is too much. We clipped it uniformely to {_clipped_volume}"
        )
    df = df.sort(["redate"])
    docs, times, raws = (
        [str(d) for d in list(df.select("silo"))[0]],
        [d for d in list(df.select("redate"))[0]],
        [str(d) for d in list(df.select(session["text"]))[0]],
    )  # TODO: add wrapper around to cast times to the same format.
    embeddings = model.encode(docs, show_progress_bar=True, device=devices[0])
    ic(f"{min(25, len(docs) // 12)}")
    _min_topic_size = min(25, len(docs) // 12) if len(docs) <= 5_000 else 122
    _botpic = botpic.fire(min_topic_size=_min_topic_size, hdbscan_model=KMeans(n_clusters=session["num_clusters"]))
    topics, probs = _botpic.fit_transform(docs, embeddings=embeddings)  # we use model under the hood
    df = df.with_column(pl.Series(topics).alias("topic"))
    session["db"] = df
    info = _botpic.get_topic_info()
    docs_per_topic = (
        pl.DataFrame({"Doc": docs, "Topic": topics, "Id": range(len(docs))})
        .groupby("Topic")
        .agg(pl.col("Doc").apply(lambda x: " ".join(x)))
    )
    tf_idf, count = pipe.c_tf_idf(
        [str(d) for d in list(docs_per_topic.select(pl.col("Doc")))[0]], m=len(docs), stopwords=IStopper()
    )
    top_n_words = pipe.extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=5)
    topics_over_time = _botpic.topics_over_time(docs, times, nr_bins=20)
    plopics = pl.from_pandas(topics_over_time)
    plopics = plopics.sort("Frequency")
    engine = BM25Okapi(processor=processor)
    engine.index([{"content": d, "timestamp": t, "raw": r, "topic_id": c} for d, t, r, c in zip(docs, times, raws, topics)])
    store[uid] = engine
    _plopics = (
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

    _plopics = {k: list(v) for k, v in _plopics.items()}

    report_filepath = Path(os.getcwd()) / ".cache" / uid / str(filename.stem + ".xlsx")
    pipe.report_overall_topics(
        keywords=top_n_words,
        info=info,
        filepath=str(report_filepath),
        plopics=_plopics,
        topk=5,
    )
    if session.get("email", None) is not None:
        pipe.send_over_email(
            attachments=[report_filepath],
            receivers=[str(session["email"])],
            subject="Анализ обращений по ключевым словам в разрезе разных тематик",
            message="Во вложении файл с аналитикой.\nС уважением, Команда корневых причин.",
            author=os.environ.get("EMAIL"),
        )
    fig = _botpic.visualize_topics_over_time(topics_over_time, top_n_topics=10)

    session["plopics"] = _botpic.visualize_topics(width=1250, height=450)
    session["tropics"] = _botpic.visualize_barchart(width=312.5, height=225, title="Topic Word Scores")

    response = [
        {
            "figure": json.loads(plotly.io.to_json(plotly_wordcloud(" ".join(docs), scale=1), pretty=True)),
            "lazy_figure_api": [
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


def view_timeseries_tropics():
    return plotly.io.to_json(session["tropics"], pretty=True)


def view_timeseries_plopics():
    return plotly.io.to_json(session["plopics"], pretty=True)


def view_representation():
    uid = str(session["uid"])
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    ic(data)
    query = data["topic_name"].strip().lower()
    query = processor(query)
    q_idx, querix = int(query[0]), query[1:]
    api = data["api"].strip().lower()
    if api != "default":  # TODO: default больше не будет
        return jsonify({"docs": [], "keywords": []})
    engine = store[uid]
    response = None
    try:
        left_date, right_date = data.get("from", None), data.get("to", None)
        # TODO: here we propagate to retrieve not only keywords but overall topics
        response = engine.retrieve_top_k(querix, topic_ids=[q_idx], left_date=left_date, right_date=right_date, top_k=250)
    except:
        return jsonify({"docs": []})

    docs = pipe.pipe_paint_docs(docs=response, querix=querix, prefix=list(prefixer))
    ic(f"Получено {len(docs)} примеров в рамках запроса по тематике {q_idx} c проставленными датами")
    if left_date is None and right_date is None:
        # Let's implement pure filtering without ranking to compare the results
        df = session["db"]
        sub = [str(d) for d in list(df.filter(pl.col("topic") == q_idx).select("silo"))[0]]
        kods = pipe.pipe_paint_kods(docs=sub, engine=engine, keyworder=iworder, left_date=left_date, right_date=right_date)
        return jsonify({"docs": docs, "keywords": kods})
    return jsonify({"docs": docs})


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


__all__ = [
    "upload",
    "iload",
    "view_timeseries",
    "view_representation",
    "snapshot",
    "search",
]
