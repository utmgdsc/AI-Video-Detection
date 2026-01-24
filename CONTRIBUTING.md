# Contributing Guide

## Branch Naming Convention

Use the format: `SCRUM-XX-short-description`

**Examples:**
- `SCRUM-10-xception`
- `SCRUM-11-mesonet`
- `SCRUM-12-efficientnet`
- `SCRUM-13-aasist`
- `SCRUM-14-dataset-selection`

## Git Workflow

### 1. Start from main

```bash
git checkout main
git pull origin main
```

### 2. Create your branch

```bash
git checkout -b SCRUM-XX-short-description
```

### 3. Make your changes

Edit files, add documentation, run experiments, etc.

### 4. Commit your changes

```bash
git add .
git commit -m "SCRUM-XX <short message>"
```

### 5. Push your branch

```bash
git push -u origin SCRUM-XX-short-description
```

### 6. Open a Pull Request

- Go to GitHub and open a PR from your branch to `main`
- Use the PR template provided

## Pull Request Rules

### One PR per topic
- One PR per Jira ticket
- One PR per model if documenting a model

### PR title format
```
SCRUM-XX: <what you did>
```

**Examples:**
- `SCRUM-10: Add XceptionNet model notes and setup guide`
- `SCRUM-14: Document dataset selection rationale`

### PR description must include
- Summary of changes
- Which docs were updated (with links)
- What you verified (ran commands, checked outputs, etc.)

## PR Checklist

Before requesting review, ensure:

- [ ] Updated model docs OR dataset docs (as applicable)
- [ ] Included links to sources (paper/repo/weights)
- [ ] Included run commands + outputs (or screenshots)
- [ ] Added screenshots to `assets/` and referenced in markdown (if any)
- [ ] No dataset files committed (check `.gitignore`)

## Questions?

Bring up any workflow questions in the weekly meeting or Slack.
