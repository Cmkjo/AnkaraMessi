"""Rebuild AnkaraMessi weights using the four licensed external DFL matches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).with_name('core')))
import dfl
import model
import motion_model as motion
import joint_model as joint
import external_context as context
import heuristic_post as post
import local_possession as local


def load_data(directory, recipe):
    data = {}
    for match in recipe['matches']:
        halves, hashes = dfl.load(directory,match)
        if any(recipe['input_sha256'].get(name)!=digest for name,digest in hashes.items()):
            raise ValueError(f'Unexpected external input for {match}')
        data[match] = []
        for half in halves:
            entries = []
            for scenario in ('clean','dropout_20261102'):
                obs = half.observed
                if scenario!='clean':
                    seed=recipe['dropout_seed']+int(hashlib.sha256((match+half.name).encode()).hexdigest()[:8],16)
                    obs=model.degrade(obs,seed)
                entries.append(dict(scenario=scenario,f=model.features(obs),raw=motion.raw_motion(obs,2),
                                    pack=context.pack(obs,np.arange(0,len(obs.ball),5))))
            clean=entries[0]
            data[match].append(dict(id=match,half=half.name,y=half.targets,
                                    f=clean['f'],raw=clean['raw'],entries=entries))
    return data


def fit_base(sequences):
    params=motion.fit([(s['f'],s['y'],s['raw']) for s in sequences])
    residual=joint.fit([(s['f'],s['y'],s['raw']) for s in sequences],params)
    return params,residual


def fit_pipeline(data, matches, recipe):
    sequences=[s for match in matches for s in data[match]]
    params,residual=fit_base(sequences)
    pseudo={}
    for held in matches:
        fit=[s for match in matches if match!=held for s in data[match]]
        teacher,head=fit_base(fit)
        for half in data[held]:
            labels,_=joint.predict(half['f'],half['raw'],teacher,head)
            labels=post.merge(labels,half['f'],25,recipe['merge'])
            pseudo[held,half['half']]=labels[::5]
    fitted=context.fit(sequences,params,residual,pseudo)
    local_sequences=[(local.local_features(e['pack'],recipe['ownership_features']),s['y'][::5])
                     for s in sequences for e in s['entries']]
    owner=local.fit(local_sequences,recipe)['team']
    def compact(head):
        return {k:head[k] for k in ('scales','coefficients')}
    return dict(params={k:params[k] for k in ('stride','fps','means','variances','transitions','initial','motion_limits')},
                joint=compact(residual),
                external_fit=dict(calibration=fitted['calibration'],context=compact(fitted['context']),
                                  control=compact(fitted['control']),dead={'multi':compact(fitted['dead']['multi'])}),
                merge=recipe['merge'],control=recipe['control'],ownership_features=recipe['ownership_features'],
                ownership_head=compact(owner),**recipe['inference_contract'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dfl-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    recipe=json.loads(Path(__file__).with_name('recipe.json').read_text())
    data=load_data(args.dfl_dir,recipe)
    result=fit_pipeline(data,recipe['matches'],recipe)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    expected=json.loads((ROOT/'ankaramessi/model_parameters.json').read_text())
    if result!=expected:
        raise ValueError('Refit differs from distributed parameters. Output retained for inspection.')
    print('Refit exactly matches the distributed parameter values.')


if __name__=='__main__':
    main()
