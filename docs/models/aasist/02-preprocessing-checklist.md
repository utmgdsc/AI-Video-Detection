# PR Checklist — Dataset Preprocessing

Each student should submit a PR this week with their dataset preprocessing code and documentation. Use this checklist to ensure everything is included.

## Required Sections in PR Description

### 1. Dataset Information
- [ ] Dataset name
- [ ] Source/download link
- [ ] Dataset size (number of videos, total GB)

### 2. Data Storage
- [ ] Where dataset is stored (local path, cloud location, mounted drive)
- [ ] How to access/download it
- [ ] Any authentication or setup needed

### 3. Data Splitting
- [ ] Train/val/test split ratio (e.g., 70/15/15)
- [ ] Total counts per split (e.g., 100 train, 20 val, 15 test)
- [ ] How the split was performed (random seed, stratified, etc.)
- [ ] Code snippet showing the split command/logic

### 4. Preprocessing Pipeline
- [ ] What preprocessing steps your model needs (e.g., face extraction, normalization, resizing)
- [ ] Code files created/modified in `backend/models/[YOUR_MODEL]/preprocessing/`
- [ ] Dependencies added to requirements.txt
- [ ] Test run command and output

### 5. Code Quality
- [ ] Code follows project structure
- [ ] No dataset files committed (only scripts)
- [ ] Paths are relative or documented for reproducibility
- [ ] Code is tested and verified to work

### 6. Documentation
- [ ] Update `docs/models/[YOUR_MODEL]/02-source-and-setup.md` with preprocessing details
- [ ] Add commands to run preprocessing
- [ ] Document any assumptions or gotchas

## PR Title Format
```
SCRUM-XX: Preprocessing pipeline for [Model Name]
```

## Example Files to Include

```
backend/models/[YOUR_MODEL]/
├── preprocessing/
│   ├── __init__.py
│   ├── preprocess.py
│   └── split_dataset.py
└── requirements.txt (updated)
```

## Verification Before Submitting

- [ ] All preprocessing scripts run without errors
- [ ] Dataset is correctly split
- [ ] Documentation is clear and complete
- [ ] No large files committed
- [ ] Ready for another team member to follow the guide and reproduce the setup
