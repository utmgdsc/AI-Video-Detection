# AI-Video-Detection

Deepfake and manipulated video detection research and project showcase repository.

## Links

**Google Drive folder:**
https://drive.google.com/drive/folders/1iD2lBPm-zB8x6PrBZRFfvWOGQkOXwx4r?usp=drive_link

**Presentation deck 1:**
https://docs.google.com/presentation/d/1j3rwMip1ntUX6QFJGsPpEAPUNxofB7W2/edit?usp=sharing&ouid=104927623345744943216&rtpof=true&sd=true

**Presentation deck 2:**
https://docs.google.com/presentation/d/1KdrAlb4Npl342EriMI3y6KL2Us3b6mHIqyuLe6Pl_7o/edit?usp=sharing

## Project Idea

See [PROJECT_IDEA.md](PROJECT_IDEA.md) for the full project description, goals, and evaluation plan.

## Repo Navigation

| Folder | Description |
|--------|-------------|
| [docs/models/](docs/models/) | Model documentation (one folder per model) |
| [docs/datasets/](docs/datasets/) | Dataset evaluation and selection docs |
| [docs/weekly-plan.md](docs/weekly-plan.md) | Project roadmap and milestone checklist |
| [docs/meeting-notes/](docs/meeting-notes/) | Project log and meeting notes |
| [docs/templates/](docs/templates/) | Templates for model notes and setup guides |

### Model Folders

- [docs/models/xception/](docs/models/xception/) — XceptionNet
- [docs/models/efficientnet/](docs/models/efficientnet/) — EfficientNet
- [docs/models/mesonet/](docs/models/mesonet/) — MesoNet
- [docs/models/aasist/](docs/models/aasist/) — AASIST (audio-based)

## Collaboration Workflow (Branches + PRs)

### Branch Naming

Use descriptive branch names:

**Examples:**
- `feature/xception-integration`
- `feature/mesonet-baseline`
- `docs/dataset-selection`
- `experiment/audio-baseline`

### PR Flow

1. Create a branch from `main`
2. Fill in your documentation
3. Open a PR to `main`
4. Get review from a contributor
5. Merge

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full git workflow and PR checklist.

## Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd AI-Video-Detection

# Create your branch
git checkout -b feature/your-topic

# Make changes, then commit
git add .
git commit -m "feat: <message>"
git push -u origin feature/your-topic

# Open a PR on GitHub
```

## Project Notes

Meeting and progress notes are in [docs/meeting-notes/](docs/meeting-notes/).
