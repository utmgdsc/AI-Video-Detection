# Selected Dataset

## Dataset Name

`AIGVDBench`

## Link

`https://github.com/LongMa-2025/AIGVDBench?tab=readme-ov-file`

## Why Chosen

- Reason 1: Multiple AI models are used to generate the fake videos
- Reason 2: Contain over 400k videos
- Reason 3: Some are generated based on real video while some are completely AI generated

## License Confirmation

- License type: `cc-by-4.0`
- Academic use allowed: `Yes`
- Any restrictions: `Attribution Required`
- Terms link: `[https://huggingface.co/datasets/AIGVDBench/AIGVDBench](https://huggingface.co/datasets/AIGVDBench/AIGVDBench)`

## Planned Splits

| Split | Size | Notes |
|-------|------|-------|
| Train | 280000 videos | 70 % |
| Validation | 60000 videos | 15% |
| Test | 60000 videos| 15% |

## Preprocessing Plan

1. Step 1: Download/locate dataset
# Path should be at: `AI-VIDEO-DETECTION/backend/dataset/AIGVDBench/AIGVDBench/`

2. Step 2: Split the dataset
at /AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench$
python3 split_videos_standard_split
or
python3 split_videos_left_out_easyAnimate.py

3. Step 3: Run preprocessing
at ~/AI-Video-Detection/backend/models/DeepFake-EfficientNet$
./extract_face_efficientNet.sh


## Risks / Limitations

- Risk 1: Preprocess 400k videos take 2-3 hours, need to use tmux to keep terminal alive
- Risk 2:
- Limitation 1:

## Download Instructions
On gdgserver
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AIGVDBench/AIGVDBench', repo_type='dataset', local_dir='./AIGVDBench')"
or if hugging face cli is installed:
huggingface-cli download --repo-type dataset <DATASET_NAME> --local-dir ./data
```

## Data Organization

```
(venv) gdgteam1@lisa:~/AI-Video-Detection/backend/dataset/AIGVDBench/AIGVDBench$ tree -L 5 -I "*.mp4"
.
├── ClosedSource
│   ├── causvid_24fps
│   ├── causvid_24fps.zip
│   ├── Gen2
│   │   └── videos
│   ├── Gen2.zip
│   ├── Gen3
│   │   └── videos
│   ├── Gen3.zip
│   ├── Jimeng
│   │   └── videos
│   ├── Jimeng.zip
│   ├── kling
│   ├── kling.zip
│   ├── Luma
│   ├── Luma.zip
│   ├── Opensora
│   ├── Opensora.zip
│   ├── pika
│   ├── pika.zip
│   ├── single
│   ├── Sora
│   ├── Sora.zip
│   ├── vidu
│   ├── vidu.zip
│   ├── wan
│   └── wan.zip
├── OpenSource
│   ├── I2V
│   │   ├── EasyAnimate
│   │   │   └── videos
│   │   ├── EasyAnimate.zip
│   │   ├── LTX
│   │   │   └── videos
│   │   ├── LTX.zip
│   │   ├── Pyramid-Flow
│   │   │   └── videos
│   │   ├── Pyramid-Flow.zip
│   │   ├── SEINE
│   │   │   └── videos
│   │   ├── SEINE.zip
│   │   ├── SVD
│   │   │   └── videos
│   │   ├── SVD.zip
│   │   ├── VideoCrafter
│   │   │   └── videos
│   │   └── VideoCrafter.zip
│   ├── T2V
│   │   ├── AccVideo
│   │   │   └── videos
│   │   ├── AccVideo.zip
│   │   ├── AnimateDiff
│   │   │   └── home
│   │   │       └── lma
│   │   ├── AnimateDiff.zip
│   │   ├── Cogvideox1.5
│   │   │   └── videos
│   │   ├── Cogvideox1.5.zip
│   │   ├── EasyAnimate
│   │   │   └── videos
│   │   ├── EasyAnimate.zip
│   │   ├── HunyuanVideo
│   │   │   └── videos
│   │   ├── HunyuanVideo.zip
│   │   ├── IPOC
│   │   │   └── videos
│   │   ├── IPOC.zip
│   │   ├── LTX
│   │   │   └── videos
│   │   ├── LTX.zip
│   │   ├── Open-Sora
│   │   │   └── videos
│   │   ├── Open-Sora.zip
│   │   ├── Pyramid-Flow
│   │   │   └── videos
│   │   ├── Pyramid-Flow.zip
│   │   ├── RepVideo
│   │   │   └── videos
│   │   ├── RepVideo.zip
│   │   ├── VideoCrafter
│   │   │   └── videos
│   │   ├── VideoCrafter.zip
│   │   ├── Wan2.1
│   │   │   └── videos
│   │   └── Wan2.1.zip
│   └── V2V
│       ├── Cogvideox1.5
│       │   └── videos
│       ├── Cogvideox1.5.zip
│       ├── LTX
│       │   └── videos
│       └── LTX.zip
├── Real
│   ├── Real.zip
│   └── videos
├── split_dataset
│   ├── dataset_easyanimate_video_not_in_train_val
│   │   ├── test
│   │   │   ├── i2v
│   │   │   ├── real
│   │   │   ├── t2v
│   │   │   └── v2v
│   │   ├── train
│   │   │   ├── i2v
│   │   │   ├── real
│   │   │   ├── t2v
│   │   │   └── v2v
│   │   └── val
│   │       ├── i2v
│   │       ├── real
│   │       ├── t2v
│   │       └── v2v
│   ├── dataset_standard_splits
│   │   ├── test
│   │   │   ├── i2v
│   │   │   ├── real
│   │   │   ├── t2v
│   │   │   └── v2v
│   │   ├── train
│   │   │   ├── i2v
│   │   │   ├── real
│   │   │   ├── t2v
│   │   │   └── v2v
│   │   └── val
│   │       ├── i2v
│   │       ├── real
│   │       ├── t2v
│   │       └── v2v
│   ├── EasyAnimate_I2V
│   │   └── videos
│   └── EasyAnimate_T2V
│       └── videos
├── split_videos_left_out_easyAnimate.py
└── split_videos_standard_split

55 directories, 34 files
```
