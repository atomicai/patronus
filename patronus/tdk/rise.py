import logging
import os
import pathlib

import dotenv
from flask import Flask, send_from_directory

from flask_session import Session
from patronus.storing.module import MemoDocStore
from patronus.tdk import prime

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

top_k = int(os.environ.get("TOP_K", 5))
index = os.environ.get("INDEX", "document")
store = MemoDocStore(index=index)  # This will change to "as service"
cache_dir = pathlib.Path(os.getcwd()) / ".cache"

app = Flask(
    __name__,
    template_folder="build",
    static_folder="build",
    root_path=pathlib.Path(os.getcwd()) / "patronus",
)
app.secret_key = "expectopatronum"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

logger.info("NEW INSTANCE is created")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


app.add_url_rule("/searching", methods=["POST"], view_func=prime.search)
app.add_url_rule("/uploading", methods=["POST"], view_func=prime.upload)
app.add_url_rule("/iloading", methods=["POST"], view_func=prime.iload)
app.add_url_rule("/downloading/<filename>", methods=["GET"], view_func=prime.download)
app.add_url_rule("/viewing", methods=["GET"], view_func=prime.view)
app.add_url_rule("/viewing_timeseries", methods=["POST"], view_func=prime.view_timeseries)
app.add_url_rule("/viewing_timeseries_plopics", methods=["POST"], view_func=prime.view_timeseries_plopics)
app.add_url_rule("/viewing_timeseries_tropics", methods=["POST"], view_func=prime.view_timeseries_tropics)
app.add_url_rule("/viewing_representation", methods=["POST"], view_func=prime.view_representation)
app.add_url_rule("/snapshotting", methods=["POST"], view_func=prime.snapshot)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777)
