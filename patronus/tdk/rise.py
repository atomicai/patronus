import logging
import os
import pathlib

import dotenv
import numpy as np
# import plotly.express as px
from flask import (Flask, jsonify, render_template, request, send_file,
                   send_from_directory, session)
from icecream import ic
from kombu import Connection, Exchange, Queue
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

from flask_session import Session
from patronus.etc import Document
from patronus.storing.module import MemoDocStore
from patronus.tdk import prime
from patronus.tooling import initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# np.random.seed(22)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

top_k = int(os.environ.get("TOP_K", 5))
index = os.environ.get("INDEX", "document")
store = MemoDocStore(index=index)
cache_dir = pathlib.Path(os.getcwd()) / ".cache"

exchange = Exchange('media', 'direct', durable=True)
inqueue = Queue('inq', exchange=exchange, routing_key="aiquery")
outqeue = Queue("ouq", exchange=exchange, routing_key="airesponse")

app = Flask(
    __name__,
    template_folder='build',
    static_folder='build',
    root_path=pathlib.Path(os.getcwd()) / 'patronus',
)
app.secret_key = "expectopatronum"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

logger.info("NEW INSTANCE is created")


@app.route('/', defaults={'path': ''})
@app.route("/<path:path>")
def index(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


app.add_url_rule("/searching", methods=["POST"], view_func=prime.search)
app.add_url_rule("/uploading", methods=["POST"], view_func=prime.upload)
app.add_url_rule("/iloading", methods=["POST"], view_func=prime.iload)
app.add_url_rule("/downloading/<filename>", methods=["GET"], view_func=prime.download)
app.add_url_rule("/viewing", methods=["GET"], view_func=prime.view)
app.add_url_rule("/viewing_timeseries", methods=["POST"], view_func=prime.view_timeseries)
app.add_url_rule("/viewing_clustering", methods=["POST"], view_func=prime.view_clustering)
app.add_url_rule("/snapshotting", methods=["POST"], view_func=prime.snapshot)
