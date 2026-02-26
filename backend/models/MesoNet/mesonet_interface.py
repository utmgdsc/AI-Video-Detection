"""
Contains a class that mimics the functionalities of a PyTorch model.

This model uses a different virtual environment in Python 3.6, which is different from the ensemble's 3.10.
In order to communicate with the model, we start a separate process that runs MesoNet with the following command:
(venv)/backend/.../MesoNet$ uvicorn mesonet_server:app --host 127.0.0.1 --port 8000

Then through the MesoNetClient class, we make HTTP requests to the server endpoints. To be more efficient,
the MesoNetClient saves information locally instead of packaging it up to be sent through a request.
In doing so, this requires mesonet_interface.py and mesonet_server.py to be placed in the same directory,
but the class can be imported anywhere.

3.6 packages: TODO
pip install fastapi==0.63.0 uvicorn==0.13.4

3.10 packages: TODO
pip install requests
"""

import subprocess
import requests
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
LOG_DIR = os.path.join(BASE_DIR, "logs")

BASE_URL = "http://"  # + "127.0.0.1:8000" to form complete URL
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# TODO: Path to Conda python.exe. Should not use abs path, but for now
HOME_PATH = "/home/gdgteam1"
ENV_PATH = HOME_PATH + "/miniconda3/envs/mesonet/bin/python3"

DEFAULT_ARCHITECTURE = "Meso4"
DEFAULT_WEIGHTS_PATH = "weights/Meso4_custom_weight1_epoch7.h5"


class MesoNetClient:

    def __init__(self):
        debug("Initializing new MesoNet Client")
        self.url = BASE_URL + f"{DEFAULT_HOST}" + ":" + f"{DEFAULT_PORT}"
        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT
        self.env_path = ENV_PATH
        self.server_process = None
        self.server_log = None

        self.ensure_server_running()

    def ensure_server_running(self):
        debug("Checking server is running...")
        try:
            debug("Sending test POST")
            response = requests.get(self.url + "/test_server", timeout=1)
            assert response.status_code == 200, "Server is not running."
            debug("Server is running.")
        except:
            print("Starting MesoNet server...")
            self.start_server()
            debug("Waiting until ready")
            self.wait_until_ready()

    def start_server(self, save_log=True):
        output = subprocess.DEVNULL
        debug("Trying to open server log")
        if save_log:
            self.server_log = open(os.path.join(LOG_DIR, "meso_server.txt"), "a")
            output = self.server_log

        debug("Trying to run server")
        self.server_process = subprocess.Popen(
            [
                self.env_path,
                "-u",
                "-m", "uvicorn",
                "mesonet_server:app",
                "--host", f"{self.host}",
                "--port", f"{self.port}"
            ],
            cwd=BASE_DIR,
            stdout=output,
            stderr=output
        )
        debug("Server started!")

    def wait_until_ready(self):
        for _ in range(20):
            try:
                debug("Testing connection...")
                response = requests.get(self.url + "/test_server", timeout=1)
                assert response.status_code == 200, "Server is not running."
                debug("Server ready.")
                return
            except:
                time.sleep(0.5)

        raise RuntimeError("Server failed to start")

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
        if self.server_log is not None and not self.server_log.closed:
            self.server_log.close()

    def __exit__(self, exc_type, exc, tb):
        self.stop_server()

    # =============== SERVER COMMUNICATION

    def load_model(self, weights_path=None):
        debug("Asking server to load model...")
        architecture = DEFAULT_ARCHITECTURE
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH

        response = requests.post(
            self.url + "/load_model",
            json={
                "architecture": architecture,
                "weights_path": weights_path
            }
        )
        debug(f"Load status: {response.status_code}")
        debug(f"Load text: {response.text}")
        if response.status_code == 200:
            debug("Model loaded successfully.")
            return self
        debug("Model failed to load.")
        # Else model failed to load
        return None
        

    def process(self, faces, stop_server=True):
        """
        Analyze faces for deepfake detection.

        Args:
            faces: List of face images (cropped from video frames)
            model_cfg: Dictionary of parameters defined in ensemble.yaml for this model

        Returns:
            dict: {
                'score': float (0-1, higher = more likely fake),
                'per_frame_scores': list of floats,
                'details': str
            }
        """
        # Save faces to npy file
        
        os.makedirs(TEMP_DIR, exist_ok=True)
        np.save(os.path.join(TEMP_DIR, "faces.npy"), faces)

        # Send npy file path
        response = requests.post(
            self.url + "/process",
            json={"faces_path": "temp/faces.npy"}
        )
        
        debug(f"Process status: {response.status_code}")
        debug(f"Process text: {response.text}")
        data = response.json()

        if stop_server:
            self.stop_server()

        if not data["success"]:
            return {}
        return data["predictions"]
    
    def cleanup(self):
        self.stop_server()


debug_num = 0


def debug(msg):
    global debug_num
    print(f"DEBUG {debug_num} =====: {msg}")
    debug_num += 1


if __name__ == "__main__":
    print("Begin testing for mesonet_interface.py")
    # We are simulating calls that would be made from from mesonet.py
    weights_path = "weights/Meso4_DF.h5"

    print("\tTesting initialization:")
    model = MesoNetClient()

    print("\tTesting load_model:")
    model.load_model(weights_path)

    print("\tStopping server:")
    model.stop_server()
    print("End")
