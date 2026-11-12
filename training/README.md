# External training and validation

Only the four DFL games listed in recipe.json are accepted. Expected hashes pin the exact dataset files. Dataset attribution and access are documented in the technical report.

    python training/download_dfl.py --output-dir /path/to/DFL
    python training/train.py --dfl-dir /path/to/DFL --output /tmp/refitted_model.json

Training reconstructs the Gaussian HMM, motion limits, logistic residuals, out-of-fold team context and local ownership head. It compares all resulting parameter values with the distributed model. It does not overwrite the release weights. The supplied recipe preserves the exact training order and synthetic dropout seed.

Nested leave-one-match-out validation excludes the outer held match from every inner teacher. It evaluates clean tracking and three fixed dropout masks.

    python training/validate.py --dfl-dir /path/to/DFL --output /tmp/dfl_validation.json

The training core is deliberately separate from release inference. Its files contain the full fit and data-adapter functions needed to inspect the provenance of the weights. The download requires network access and verifies each dataset file against its recorded SHA-256. No DFL or match data should be committed.

Optional PFF transfer check, using an independently obtained authorized data copy with tracking, metadata and roster files for games 3812–3816:

    python training/validate_pff.py --pff-dir /path/to/FifaWorldCup --output /tmp/pff_validation.json

The PFF event-derived reference is used only for scoring. Temporary aliases handle the metadata and roster filenames without changing the source data.
