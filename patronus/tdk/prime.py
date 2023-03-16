import collections
import json
import logging
import os
import uuid
from pathlib import Path

import plotly
import polars as pl
import pyarrow.parquet as pq
import torch
from bertopic import BERTopic
from flask import jsonify, request, send_file, session
from icecream import ic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from werkzeug.utils import secure_filename

from patronus.modeling.module import BM25Okapi
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


model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=devices[0])
store = collections.defaultdict()
botpic = BERTopic(
    language="multilingual", n_gram_range=(1, 2), min_topic_size=3, umap_model=UMAP(random_state=42), embedding_model=model
)
stopper = IStopper()
prefixer = IPrefixer()


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


# "title": str vs
# "label": str
"text"
# "doc_score" -> str
"timestamp"
# "highlight" -> [{"lo": 1, "hi": 5, "score": "abracadabra", "color": 1}]
# "highlight_idx" -> 0


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
    text_column, datetime_column = data.get("text", None), data.get("datetime", None)
    is_text_ok, is_date_ok = False, False
    df = pl.scan_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    if text_column is not None and text_column in df.columns:
        session["text"] = text_column
        is_text_ok = True
    if datetime_column is not None and datetime_column in df.columns:
        session["datetime"] = datetime_column
        is_date_ok = True

    session["email"] = data.get("email", None)

    if is_date_ok and is_text_ok:
        return jsonify({"success": "In progress to the moon🚀"})
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
    # fname = Path(filename)
    # TODO: Push here to the queue
    # Here we fetch from queue (BROKER) until the required message is receieved

    dfr = pl.read_parquet(cache_dir / uid / f"{filename.stem}.parquet")  # raw without cleaning, etc..
    tecol, dacol = session["text"], session["datetime"]
    dfr = (
        dfr.with_columns([pl.col(tecol).is_null().alias("is_empty")])
        .filter(~pl.col("is_empty"))
        .select([pl.col(tecol), pl.col(dacol)])
    )
    try:
        dfr = dfr.sort([dacol])
    except:  # add logging to determine futher the problem.
        ic(f"Failed to sort by datetime file {uid}-{filename.stem}")
    # <--->
    dfs = ppipe.pipe_polar(
        dfr, txt_col_name=tecol, fn=stopper, seps=[":", " "]
    )  # dataframe to which `IStopper` had been applied
    ic("Stopwords removal is completed")
    dfs = (
        dfs.with_columns([pl.col(tecol).apply(ppipe.pipe_nullifier).alias("is_empty")])
        .filter(~pl.col("is_empty"))
        .select([pl.col(tecol), pl.col(dacol)])
    )
    ic(f"Final size after preprocessing is {str(dfs.shape)}")
    # Maybe there is a processed file already?
    # df = next(get_data(data_dir=cache_dir / uid, filename=fname.stem, ext=fname.suffix))
    # docs, times = df["text"].tolist(), df["datetime"].tolist()
    docs, times, raws = (
        [str(d) for d in list(dfs.select(session["text"]))[0]],
        [str(d) for d in list(dfs.select(session["datetime"]))[0]],
        [str(d) for d in list(dfr.select(session["text"]))[0]],
    )

    embeddings = model.encode(docs, show_progress_bar=True, device=devices[0])
    topics, probs = botpic.fit_transform(docs, embeddings=embeddings)  # we use model under the hood
    info = botpic.get_topic_info()
    docs_per_topic = (
        pl.DataFrame({"Doc": docs, "Topic": topics, "Id": range(len(docs))})
        .groupby("Topic")
        .agg(pl.col("Doc").apply(lambda x: " ".join(x)))
    )
    tf_idf, count = pipe.c_tf_idf(
        [str(d) for d in list(docs_per_topic.select(pl.col("Doc")))[0]], m=len(docs), stopwords=stopper
    )
    top_n_words = pipe.extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=5)
    # Мы показываем польтзователю,, что идет индексация и что-то считается.
    topics_over_time = botpic.topics_over_time(docs, times, nr_bins=20)
    plopics = pl.from_pandas(topics_over_time)
    plopics = plopics.sort("Frequency")
    engine = BM25Okapi(processor=processor)
    engine.index([{"content": d, "timestamp": t, "raw": r, "topic_id": c} for d, t, r, c in zip(docs, times, raws, topics)])
    store[uid] = engine
    list(plopics["Timestamp"].unique())
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

    # TODO: use timestamp from topics_over_time!
    report_filepath = Path(os.getcwd()) / ".cache" / uid / str(filename.stem + ".xlsx")
    pipe.report_overall_topics(
        keywords=top_n_words,
        info=info,
        filepath=str(report_filepath),
        plopics=_plopics,
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
    fig = botpic.visualize_topics_over_time(topics_over_time, top_n_topics=10)

    # TODO:
    session["plopics"] = botpic.visualize_topics(width=1250, height=450)
    session["examples"] = botpic.visualize_documents(raws, width=1250, height=450, embeddings=embeddings)
    session["tropics"] = botpic.visualize_barchart(width=312.5, height=225, title="Topic Word Scores")

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
        response = engine.retrieve_top_k(querix, topic_ids=[q_idx], top_k=100)
    except:
        return jsonify({"docs": []})

    docs = pipe.pipe_paint_docs(docs=response, querix=querix, prefix=list(prefixer))
    kods = pipe.pipe_paint_kods(querix=querix, engine=engine)

    return jsonify({"docs": docs, "keywords": kods})


def view_representation_keywords():
    uid = str(session["uid"])
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    query = data["topic_name"].strip().lower()
    query = processor(query)
    q_idx, querix = int(query[0]), query[1:]
    ic(data)
    engine = store[uid]
    try:
        response = engine.retrieve_top_k(querix, topic_ids=[q_idx], top_k=100)
    except:
        return jsonify({"docs": []})
    kods = pipe.pipe_paint_kods(querix=querix, engine=engine)

    return jsonify(kods)


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
    "view_representation_keywords",
    "snapshot",
    "search",
]
