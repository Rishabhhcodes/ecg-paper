# Team workflow

Five people, one repo, one paper. Each stage is contributed by its owner,
on its own branch, **after they have run and tested it**.

## One-time setup (everyone)

```bash
git clone <repo-url>
cd ecg-paper

git config user.name  "Your Name"
git config user.email "your@email.com"

python -m venv venv
source venv/bin/activate        # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How a stage gets contributed

The point of this repo is that the history shows who did what. So the
person who owns a stage is the person who commits it.

```bash
git checkout main
git pull                                   # always, before branching
git checkout -b stage2/model-grid          # your stage's branch
```

Add your files, **run them, break them, fix them**. Only once the stage
works on your machine:

```bash
git add src/run_grid.py src/models.py ...
git commit -m "stage2: model grid with GroupKFold and four imbalance strategies"
git push -u origin stage2/model-grid
```

Then open a Pull Request and ask a teammate to review.

Do not commit code you have not run. It shows the moment someone asks
you a question about it — and the whole reason for this workflow is that
each of us can defend our own section.

| Branch | Stage |
|---|---|
| `stage1/data-pipeline` | Data preparation |
| `stage2/model-grid` | Model training |
| `stage3/decision-layer` | Calibration, thresholds, reject option |
| `stage4/evaluation` | DS2 final run and metrics |
| `stage5/explainability` | SHAP segments and faithfulness |
| `paper/draft` | Writing |

Because each stage owns different files, conflicts are rare by design.
The one shared file is `src/config.py` — see below.

## Never commit data

`data/` and `artifacts/` are in `.gitignore` and must stay there.

MIT-BIH is ~90 MB; `ds1.npz` and `ds2.npz` are ~90 MB more. Git stores a
full copy of every version of a binary file, so committing them once
makes the repo permanently slow to clone, and two people regenerating
them produces a conflict git cannot merge.

The build is deterministic — same records, same code, same output on
every machine. **Git tracks the recipe, not the cake.**

To share a `.npz`, use the shared Drive folder:

```
Drive/ecg-paper/
├── stage1-data/          ds1.npz, ds2.npz, manifest.json
└── stage2-predictions/   preds/ (~10 MB)
```

## Daily rhythm

```bash
git pull origin main            # start of every session
# ... work ...
git add <your files>
git commit -m "stage2: cap ADASYN at 35% of majority"
git push
```

Commit when something works, not at the end of the day. A commit that
runs is a restore point; a commit that does not is noise.

Message format: `stage<N>: what changed`.
Good — `stage2: cap ADASYN at 35% of majority`.
Useless — `update`, `fix`, `changes`.

## Reviewing a PR

The reviewer's job is **not** to admire the code. It is to pull the
branch and run it on their own machine. If it only works on the author's
laptop, it is not done — and you want to discover that in October, not
December.

## `src/config.py` is shared — treat it carefully

Every stage imports it. The record lists, the AAMI mapping, the window
size and the evaluation classes live there.

Changing any of those invalidates all existing results. So:

1. Raise it with the team first
2. Change it in its own commit, alone
3. Say so in the PR title: `config: change window to +-90 samples`
4. Everyone deletes `artifacts/` and rebuilds

## Repo layout

```
ecg-paper/
├── src/              code — this is what git tracks
├── docs/             per-stage instructions
├── data/             MIT-BIH, gitignored
├── artifacts/        ds1.npz, ds2.npz, predictions — gitignored
├── requirements.txt
└── README.md
```

Notebooks are fine for exploring, but anything that produces a number in
the paper must be a script in `src/` — a notebook's output depends on
the order somebody happened to run the cells in.
