"""Reproduce nested external leave-one-match-out validation, with fixed dropout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import train

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'evaluation'))
from ankaramessi.solution import predict_half
import possession_metrics as metrics


def score(y,pred):
    gt,p=metrics.label_codes(y),metrics.label_codes(pred)
    cm=metrics.confusion(gt,p)[:3]
    macro=float(np.mean([2*cm[i,i]/(cm[i].sum()+cm[:,i].sum()) for i in range(3)]))
    runs=metrics.valid_runs(gt)
    spell=metrics.spell_scores(gt,metrics.fill_abstentions(p,runs),runs)
    return dict(spell_f1=spell['mean_f1'],macro_f1=macro,
                prediction_sha256=hashlib.sha256(''.join(pred).encode()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dfl-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    recipe=json.loads(Path(__file__).with_name('recipe.json').read_text())
    data=train.load_data(args.dfl_dir,recipe)
    rows=[]
    for held in recipe['matches']:
        fit=[m for m in recipe['matches'] if m!=held]
        weights=train.fit_pipeline(data,fit,recipe)
        halves,_=train.dfl.load(args.dfl_dir,held)
        for half in halves:
            for seed in (None,20261102,20261103,20261104):
                obs=half.observed
                if seed is not None:
                    derived=seed+int(hashlib.sha256((held+half.name).encode()).hexdigest()[:8],16)
                    obs=train.model.degrade(obs,derived)
                pred=predict_half(obs,weights)
                rows.append(dict(match=held,half=half.name,scenario='clean' if seed is None else f'dropout_{seed}',
                                 fit_matches=fit,**score(half.targets,pred)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol='Nested external LOMO',results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
