import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

from classifiers import *


class LoadModel(BaseModel):
    architecture: str
    weights_path: str


class Process(BaseModel):
    faces_path: str


# Global Variables
app = FastAPI()

model = None
graph = None

ARCHITECTURE_MAP = {
    "Meso1": Meso1,
    "Meso4": Meso4,
    "MesoInception4": MesoInception4
}


@app.get("/test_server")
def test_server():
    return {"success": True}


@app.post("/load_model")
def load_model(data: LoadModel):
    global model, graph

    # Every time we load a new model, we must clear any potential previous models
    tf.keras.backend.clear_session()

    if data.architecture not in ARCHITECTURE_MAP:
        return {"success": False,
                "error": "Unknown architecture. Check ensemble.yaml or mesonet_server.py."}
    model = ARCHITECTURE_MAP[data.architecture]()

    model.load(data.weights_path)
    graph = tf.get_default_graph()

    return {"success": True}


@app.post("/process")
def process(data: Process):
    global model, graph
    images = []

    if model is None or graph is None:
        return {"success": False,
                "error": "No model loaded."}

    images = np.load(data.faces_path)

    # Normalize the images, if not already normalized
    if images.max() > 1.0:
        images = images / 255.0

    with graph.as_default():
        preds = model.predict(images).tolist()

    return {"success": True,
            "predictions": preds}
