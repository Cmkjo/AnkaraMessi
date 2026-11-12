"""Evaluate the frozen model on external PFF smoothed games, without fitting."""
import argparse
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import train
from validate import score

sys.path.insert(0,str(train.ROOT))
from ankaramessi import context, geometry, hmm, kinematics, ownership, residual, segments
import pff_reader as pff
import pff_smoothed as smoothed


def predict(obs,settings):
    params=copy.deepcopy(settings['params'])
    params.update(stride=1,fps=5)
    fitted=settings['external_fit']
    f=geometry.features(obs)
    raw=kinematics.raw_motion(obs,2)
    index,hold=pff.grid_indices(len(f),obs.fps)
    pack=context.pack(obs,index)
    emission=residual.frozen_emissions(f[index],raw[index],params,settings['joint'])
    signals=context.features(pack,fitted['calibration'],params)
    shape=context.predict_logit(signals['context'],fitted['context'])
    shape[~np.isfinite(signals['context']).all(axis=1)]=0
    odds=np.clip(shape,-3,3)*(1-signals['quality'].max(axis=1))
    odds+=np.clip(context.predict_logit(signals['control'],fitted['control']),-3,3)
    change=kinematics.control_change(pack,fitted['calibration'],params['motion_limits'],settings['control'])
    emission=hmm.fuse(emission,np.clip(odds+change,-3,3))
    emission[:,2]+=np.clip(context.predict_logit(signals['multi'],fitted['dead']['multi']),-8,8)
    labels=hmm.decode(emission,params,len(index))[hold]
    base=segments.merge(labels,f,obs.fps,settings['merge'])
    return ownership.refine(base,hold,emission,ownership.local_features(pack,settings['ownership_features']),
                            params,settings['ownership_head'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pff-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads((train.ROOT/'ankaramessi/model_parameters.json').read_text())
    rows=[]
    with TemporaryDirectory() as stage:
        root=Path(stage)
        for match in ('3812','3813','3814','3815','3816'):
            source=args.pff_dir/match
            target=root/match
            target.mkdir()
            roles={f'{match}.json':[f'{match}.json',f'Metadata_{match}.json'],
                   f'Roster_{match}.json':[f'Roster_{match}.json',f'Rosters_{match}.json'],
                   f'{match}.jsonl':[f'{match}.jsonl']}
            for name,choices in roles.items():
                found=[source/c for c in choices if (source/c).is_file()]
                if len(found)!=1:
                    raise ValueError(f'Ambiguous or missing {name}')
                (target/name).symlink_to(found[0].resolve())
            raw,_=pff.load(root,match)
            parts=smoothed.load(root,match,raw)
            for part in parts:
                obs=pff.observations(part,'all_finite')
                rows.append(dict(match=match,half=part.number,**score(part.reference,predict(obs,settings))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(reference='Event-derived PFF, smoothed all-finite',results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
