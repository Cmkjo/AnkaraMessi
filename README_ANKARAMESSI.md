# AnkaraMessi — Reproducible possession inference

Team members: Matteo Tognetti, Carlo Borghini, Pietro Borracelli and Christian Petani.

## Install and run

Verified environment: Python 3.14.3, NumPy 2.4.4 and pandas 3.1.0. From the repository root:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-ankaramessi.txt
    python -B -m ankaramessi.solution --tactical-dir data/tactical
    python evaluation/validate_submission.py evaluation/results/AnkaraMessi.csv

The input directory must contain BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt. The command also accepts an external absolute path. Default output is evaluation/results/AnkaraMessi.csv. The expected file hash is:

    8d31ea61c78d059de2bff86116ab1692727c30ff5eb588a058ca105d92cdcbdc

Inference interprets the numerical player speeds as m/s following the official data description, while checking the XML structure. File hashes document the audited release export. A different export still needs the same supported layout, channel structure and 25 Hz cadence.

## Inspect and reproduce training

    python training/download_dfl.py --output-dir /path/to/DFL
    python training/train.py --dfl-dir /path/to/DFL --output /tmp/refitted_model.json
    python training/validate.py --dfl-dir /path/to/DFL --output /tmp/dfl_validation.json

The fit uses four external DFL games, never the target-game labels. Refit compares every parameter value with the release. Nested LOMO excludes the outer match from every teacher. [Training details](training/README.md)

Optional PFF transfer check after obtaining authorized data from the provider:

    python training/validate_pff.py --pff-dir /path/to/FifaWorldCup --output /tmp/pff_validation.json

PFF is optional for the transfer check. Request access through the provider’s release form, follow its download instructions and retain any terms received during access. Data files are excluded from the repository. [Provider access](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)

## Documentation and integrity

- [English technical report](docs/AnkaraMessi.md)
- [Release manifest](docs/release_manifest.json)

    shasum -a 256 -c SHA256SUMS_ANKARAMESSI.txt

The report's public-window figure is included under docs/figures. It contains labels only. To regenerate the local original after inference:

    python tools/plot_public_window.py --public-gt evaluation/gt/BAR-ATH_possession_gt_public.csv

Commit source, fitted parameters, training recipe and documentation. Keep tracking, video, external datasets, local CSV outputs, caches and logs excluded. Follow the organisers' SUBMISSION.md for the CSV, public repository URL and technical document. Commit and publication remain the team's responsibility.
