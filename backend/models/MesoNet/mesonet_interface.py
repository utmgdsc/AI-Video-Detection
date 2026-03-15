"""
Contains a class that mimics the functionalities of a PyTorch model.

This model uses a different virtual environment in Python 3.6, which is different from the ensemble's 3.10.
In order to communicate with the model, we start a separate process that runs MesoNet with the following command:
(venv)/backend/.../MesoNet$ uvicorn mesonet_server:app --host 127.0.0.1 --port 8000

Then through the MesoNetClient class, we make HTTP requests to the server endpoints. To be more efficient,
the MesoNetClient saves information locally instead of packaging it up to be sent through a request.
In doing so, this requires mesonet_interface.py and mesonet_server.py to be placed in the same directory,
but the class can be imported anywhere.
"""

import subprocess
import requests
import numpy as np
import time
import os
import warnings

PRINT_DEBUG = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
LOG_DIR = os.path.join(BASE_DIR, "logs")

BASE_URL = "http://"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

DEFAULT_ARCHITECTURE = "Meso4"
DEFAULT_WEIGHTS_PATH = "weights/Meso4_custom_weight1_epoch7.h5"

class MesoNetClient:

    def __init__(self, cfg):
        debug("Initializing new MesoNet Client")
        self.host = DEFAULT_HOST
        self.load_config(cfg)
        self.url = BASE_URL + f"{DEFAULT_HOST}" + ":" + f"{self.port}"
        
        faces_save_file_name = f"faces-port-{self.port}.npy"
        self.faces_save_path = os.path.join(TEMP_DIR, faces_save_file_name)
        
        self.server_process = None
        self.server_log = None

        self.ensure_server_running()
    
    def load_config(self, cfg):
        self.env_path = None
        self.port = DEFAULT_PORT
        self.architecture = DEFAULT_ARCHITECTURE
        self.weights_path = DEFAULT_WEIGHTS_PATH
        
        if "env_path" not in cfg:
            raise AttributeError("MesoNet env_path not found. See ensemble.yaml MesoNet env_path.")
        self.env_path = cfg["env_path"]
        
        if "port" not in cfg:
            warnings.warn(f"Port not found in ensemble.yaml. Using default '{self.port}'.", UserWarning)
        else:
            self.port = cfg["port"]
            
        if "architecture" in cfg:
            self.architecture = cfg["architecture"]
            
        if "weights_path" in cfg:
            self.weights_path = cfg["weights_path"]
        print(f"Using meso weight {self.weights_path}.")
        

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
            debug("Waiting until ready...")
            self.wait_until_ready()

    def start_server(self, save_log=False):
        output = subprocess.DEVNULL
        if save_log:
            debug("Trying to open server log")
            os.makedirs(LOG_DIR, exist_ok=True)
            self.server_log = open(os.path.join(LOG_DIR, f"meso_server-port-{self.port}.txt"), "a")
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
        debug("Server starting...")

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
        
        # (Low priority) DEFAULT_WEIGHT -> YAML config['weights_path'] -> weights_path (High priority)
        if weights_path is None:
            weights_path = self.weights_path

        response = requests.post(
            self.url + "/load_model",
            json={
                "architecture": self.architecture,
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
        

    def process(self, faces, stop_server=False):
        """
        Analyze faces for deepfake detection.

        Args:
            faces: List of face images (cropped from video frames)

        Returns:
            dict: {
                'score': float (0-1, higher = more likely fake),
                'per_frame_scores': list of floats,
                'details': str
            }
        """
        # Start a new server if the initial server was stopped
        self.ensure_server_running()
        
        # Save faces to npy file
        os.makedirs(TEMP_DIR, exist_ok=True)
        np.save(self.faces_save_path, faces)

        # Send npy file path
        response = requests.post(
            self.url + "/process",
            json={"faces_path": self.faces_save_path}
        )
        
        debug(f"Process status: {response.status_code}")
        debug(f"Process text: {response.text}")
        data = response.json()

        if stop_server:
            self.stop_server()

        if not data["success"]:
            return None
        return data["predictions"]
    
    def cleanup(self):
        self.stop_server()


debug_num = 0


def debug(msg):
    if PRINT_DEBUG:
        global debug_num
        print(f"DEBUG {debug_num} =====: {msg}")
        debug_num += 1
