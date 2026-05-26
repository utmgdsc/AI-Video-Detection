"""
Modal deployment for the AI Video Detection backend.

Deploy:    modal deploy modal_app.py
Dev/test:  modal serve modal_app.py   (hot-reload, temporary URL)

One-time weight upload (run from project root after downloading from GDG server):
    modal volume put deepfake-weights ./local/efficientnet.pth  /efficientnet.pth
    modal volume put deepfake-weights ./local/xceptionnet.pkl   /xceptionnet.pkl
    modal volume put deepfake-weights ./local/mesonet.h5        /mesonet.h5
    modal volume put deepfake-weights ./local/aasist.pth        /aasist.pth
"""

import modal

# ─── Container image ──────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsndfile1",
    )
    .pip_install(
        # PyTorch – Modal's CUDA drivers are pre-installed on GPU containers
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        # TensorFlow for MesoNet (TF1 compat mode via tf.compat.v1)
        "tensorflow>=2.16.0",
        # Vision
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "facenet-pytorch>=2.5.3",
        "albumentations>=1.3.0",
        "efficientnet_pytorch",
        "timm>=0.9.0",
        # Audio
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        # Numerics / ML
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        # Plotting — pulled in transitively by deepfake_detector.utils.visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # Progress bars — pulled in by deepfake_detector.utils.logger
        "tqdm>=4.65.0",
        # HTTP client — used by mesonet_interface to talk to MesoNet subprocess
        "requests>=2.31.0",
        # Web
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        # Config
        "PyYAML==6.0.1",
    )
    # Bake source code into image at deploy time
    .add_local_dir("backend", "/app/backend", copy=True)
    .workdir("/app")
)

# ─── Persistent volume for model weights ──────────────────────────────────────
# Created automatically on first deploy. Upload weights once (see header above).

weights_volume = modal.Volume.from_name("deepfake-weights", create_if_missing=True)

# ─── App ──────────────────────────────────────────────────────────────────────

app = modal.App("deepfake-detector")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/weights": weights_volume},
    timeout=300,          # max 5 min per request (long video safety net)
    memory=16384,         # 16 GB RAM
    scaledown_window=300,          # keep warm 5 min after last request
    env={
        "AIVD_CONFIG_PATH": "/app/backend/config/ensemble.yaml",
        "AIVD_DEVICE": "cuda",
        # Skip dataset path validation — training datasets are not on the server
        "AIVD_VALIDATE_PATHS": "false",
        # Weight paths point to the Modal Volume mount
        "AIVD_EFFICIENTNET_WEIGHTS_PATH": "/weights/efficientnet.pth",
        "AIVD_XCEPTION_WEIGHTS_PATH": "/weights/xceptionnet.pkl",
        "AIVD_MESONET_WEIGHTS_PATH": "/weights/mesonet.h5",
        "AIVD_AASIST_WEIGHTS_PATH": "/weights/aasist.pth",
        # MesoNet subprocess uses the same Python as the container
        "AIVD_MESONET_ENV_PATH": "/usr/bin/python3",
    },
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from backend.api.app import app as _app
    return _app
