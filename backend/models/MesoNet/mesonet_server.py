import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

from classifiers import *

PRINT_DEBUGS = False

class LoadModel(BaseModel):
    architecture: str
    weights_path: str


class Process(BaseModel):
    faces_path: str


# Global Variables
app = FastAPI()

model = None
graph = None
loaded_architecture = None
loaded_weights_path = None

ARCHITECTURE_MAP = {
    "Meso1": Meso1,
    "Meso4": Meso4,
    "MesoInception4": MesoInception4
}


@app.get("/test_server")
def test_server():
    debug("Writing test message to log (from mesonet_interface.py test_server(), expected in logs/meso_server.txt)")
    return {"success": True}


@app.post("/load_model")
def load_model(data: LoadModel):
    """
    data: {
        architecture: string,
        weights_path: string
        }
    """
    global model, graph, loaded_architecture, loaded_weights_path

    # Reuse already loaded model to avoid TensorFlow 1.x graph/thread reload issues.
    if (
        model is not None
        and graph is not None
        and loaded_architecture == data.architecture
        and loaded_weights_path == data.weights_path
    ):
        return {"success": True, "reused": True}

    debug("Clearing previous model session (no affect if no models were loaded before).")
    # Every time we load a new model, we must clear any potential previous models
    tf.keras.backend.clear_session()

    if data.architecture not in ARCHITECTURE_MAP:
        debug(f"Unknown architecture: '{data.architecture}'")
        return {"success": False,
                "error": "Unknown architecture. Check ensemble.yaml or mesonet_server.py."}
    debug(f"Selecting architecture: '{data.architecture}'")
    model = ARCHITECTURE_MAP[data.architecture]()

    debug(f"Loading weight on path: '{data.weights_path}'")
    try:
        model.load(data.weights_path)
        graph = tf.get_default_graph()
        loaded_architecture = data.architecture
        loaded_weights_path = data.weights_path
    except Exception as exc:
        # Keep API contract stable for caller and avoid hard 500 crashes.
        model = None
        graph = None
        loaded_architecture = None
        loaded_weights_path = None
        return {"success": False, "error": str(exc)}

    debug(f"MODEL SUCCESSFULLY LOADED.")
    return {"success": True}


@app.post("/process")
def process(data: Process):
    global model, graph
    images = []

    if model is None or graph is None:
        debug(f"Error: no model loaded.")
        return {"success": False,
                "error": "No model loaded."}

    debug(f"Loading images from faces file: '{data.faces_path}'")
    images = np.load(data.faces_path)

    debug(f"Normalizing images, if not already normalized.")
    # Normalize the images, if not already normalized
    if images.max() > 1.0:
        images = images / 255.0

    with graph.as_default():
        debug(f"BEGIN MAKING PREDICTIONS...")
        preds = model.predict(images).tolist()

    debug(f"PREDICTIONS MADE, RETURNING RESULTS AS:")
    debug(f"{preds}")

    return {"success": True,
            "predictions": preds}


debug_num = 0


def debug(msg):
    if PRINT_DEBUGS:
        global debug_num
        print(f"MesoServer {debug_num} =====: {msg}")
        debug_num += 1
