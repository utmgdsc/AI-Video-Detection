import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from keras import backend as K
import uuid
import threading

from classifiers import *


class Uid(BaseModel):
    uid: str


class LoadModel(BaseModel):
    uid: str
    architecture: str
    weights_path: str


class Process(BaseModel):
    uid: str
    faces_path: str


# Global Variables
app = FastAPI()

assign_uid_lock = threading.Lock()

models = {}
graphs = {}
sessions = {}
locks = {}  # To prevent concurrency issues when calling process on the same model

ARCHITECTURE_MAP = {
    "Meso1": Meso1,
    "Meso4": Meso4,
    "MesoInception4": MesoInception4
}


@app.get("/test_server")
def test_server():
    debug("Writing test message to log (from mesonet_interface.py test_server(), expected in logs/meso_server.txt)")
    return {"success": True}


@app.get("/assign_uid")
def assign_uid():
    global models, graphs, sessions, locks

    debug("Assigning uid for new model.")
    with assign_uid_lock:
        uid = str(uuid.uuid4())
        models[uid] = None
        graphs[uid] = None
        sessions[uid] = None
        locks[uid] = threading.Lock()
    return {"uid": uid}


@app.post("/load_model")
def load_model(data: LoadModel):
    """
    data: {
        uid: string,
        architecture: string,
        weights_path: string
        }
    """
    global models, graphs, sessions
    uid = data.uid
    if uid not in models:
        return {"success": False,
                "error": f"UID {uid} is not recognized."}

    if data.architecture not in ARCHITECTURE_MAP:
        debug(f"Unknown architecture: '{data.architecture}'")
        return {"success": False,
                "error": "Unknown architecture. Check ensemble.yaml or mesonet_server.py."}
    debug(f"Selecting architecture: '{data.architecture}'")
    try:
        model = ARCHITECTURE_MAP[data.architecture]()

        debug(f"Loading weight on path: '{data.weights_path}'")
        model.load(data.weights_path)
        graph = tf.get_default_graph()
        session = K.get_session()

        models[uid] = model
        graphs[uid] = graph
        sessions[uid] = session

        debug(f"MODEL SUCCESSFULLY LOADED.")
        return {"success": True,
                "uid": data.uid}
    except Exception:
        debug(f"Model '{uid}' failed to load.")
        return {"success": False,
                "error": "Model failed to load."}


@app.post("/process")
def process(data: Process):
    global models, graphs, sessions, locks
    uid = data.uid
    if uid not in models:
        return {"success": False,
                "error": f"UID {uid} is not recognized."}
    model = models[uid]
    graph = graphs[uid]
    session = sessions[uid]
    lock = locks[uid]

    images = []

    if model is None or graph is None or session is None:
        debug(f"Error: no model loaded.")
        return {"success": False,
                "error": "No model loaded."}

    debug(f"Loading images from faces file: '{data.faces_path}'")
    try:
        images = np.load(data.faces_path, allow_pickle=False)
    except:
        return {"success": False,
                "error": f"Failed to load faces from {data.faces_path}."}

    try:
        debug(f"Normalizing images, if not already normalized.")
        # Normalize the images, if not already normalized
        if images.max() > 1.0:
            images = images / 255.0

        with lock:
            with graph.as_default():
                with session.as_default():
                    debug(f"BEGIN MAKING PREDICTIONS...")
                    preds = model.predict(images).tolist()

    except:
        debug(f"PREDICTIONS FAILED.")
        return {"success": False,
                "error": f"Failed to make predictions."}

    debug(f"PREDICTIONS MADE, RETURNING RESULTS AS:")
    debug(f"{preds}")

    # TODO: Save preds as a file or send over as HTTP request
    return {"success": True,
            "predictions": preds}


@app.post("/cleanup")
def cleanup(data: Uid):
    try:
        uid = data.uid

        if uid not in models:
            return {
                "success": False,
                "error": f"UID {uid} is not recognized."
            }

        lock = locks[uid]

        with lock:
            models.pop(uid, None)
            graphs.pop(uid, None)
            sessions.pop(uid, None)
            locks.pop(uid, None)

        return {"success": True}
    except KeyError:
        return {"success": False}


debug_num = 0


def debug(msg):
    global debug_num
    print(f"SERVER {debug_num} =====: {msg}")
    debug_num += 1
