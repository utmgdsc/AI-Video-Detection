"""
Contains a class that mimics the functionalities of a PyTorch model.

3.6 packages: TODO
pip install fastapi==0.63.0 uvicorn==0.13.4

3.10 packages: TODO
pip install requests
uvicorn mesonet_server:app --host 127.0.0.1 --port 8000
"""

import subprocess
import requests
import numpy as np
import time

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
        self.url = BASE_URL + f"{DEFAULT_HOST}" + ":" + f"{DEFAULT_PORT}"
        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT
        self.env_path = ENV_PATH
        self.server_process = None
        self.server_log = None

        self.ensure_server_running()

    def ensure_server_running(self):
        try:
            response = requests.get(self.url + "/test_server", timeout=1)
            assert response.json()["success"], "Server is not running."
        except:
            print("Starting MesoNet server...")
            self.start_server()
            debug("Waiting until ready")
            self.wait_until_ready()

    def start_server(self, save_log=False):
        output = subprocess.DEVNULL
        debug("Trying to open server log")
        if save_log:
            self.server_log = open("logs/meso_server.txt", "a")
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
            stdout=output,
            stderr=output
        )
        debug("Server started!")

    def wait_until_ready(self):
        for _ in range(20):
            try:
                response = requests.get(self.url + "/test_server", timeout=1)
                assert response.json()["success"], "Server is not running."
                print("Server ready.")
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
        if response.json()["success"]:
            return self
        # Else model failed to load
        return None

    def process(self, faces, stop_server=True):
        """
        Analyze faces for deepfake detection.

        Args:
            faces: List of face images (cropped from video frames)
            model_cfg: Dictionary of parameters defined in ensemble.yaml for this model

        Returns: TODO: Change the return type? Or format correctly?
            dict: {
                'score': float (0-1, higher = more likely fake),
                'per_frame_scores': list of floats,
                'details': str
            }
        """
        # Save faces to npy file
        np.save("temp/faces.npy", faces)

        # Send npy file path
        response = requests.post(
            self.url + "/process",
            json={"faces_path": "temp/faces.npy"}
        )
        data = response.json()

        if stop_server:
            self.stop_server()

        if not data["success"]:
            return {}
        return data["predictions"]


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
