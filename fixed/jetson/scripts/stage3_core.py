"""Fixed causal features, weighted ridge and metrics for the frozen Stage 3 protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_stage2 import selected_frame
from prepare_stage2 import ROOT, RAILS, dump

OUT=ROOT/'reports/stage3'
WORK=ROOT/'derived/stage3'
MODELS=ROOT/'models/stage3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration():
    frozen=json.loads((ROOT/'manifests/stage3-freeze.json').read_text())
    for path,digest in frozen['files'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError('Frozen protocol/input changed: '+path)
    cfg=json.loads((ROOT/'manifests/stage3-config.json').read_text())
    return cfg,dict(protocol_sha256=frozen['files']['STAGE3_PROTOCOL.md'],
                    config_sha256=frozen['files']['manifests/stage3-config.json'],
                    freeze_sha256=sha(ROOT/'manifests/stage3-freeze.json'))


def feature_names(cfg):
    return cfg['temperature_features']+cfg['telemetry_extra_features']


def load_windows(cfg):
    windows=json.loads((ROOT/'manifests/windows.json').read_text())
    chosen=[w for w in windows if w['inclusion_status']=='candidate']
    assert {w['recording_id'] for w in chosen}==set(cfg['development_windows']+cfg['heldout_windows'])
    canonical={x['recording_id']:x for x in json.loads((ROOT/'manifests/canonical-files.json').read_text())}
    for w in chosen:
        if sha(ROOT/canonical[w['recording_id']]['path'])!=canonical[w['recording_id']]['sha256']:
            raise ValueError('Canonical integrity failure: '+w['recording_id'])
        expected=cfg['development_groups'] if w['recording_id'] in cfg['development_windows'] else cfg['heldout_groups']
        assert w['proposed_validation_group'] in expected
    return chosen


def arrays(frame):
    """Load once, preserving observed rows and missing off-core loads."""
    return dict(times=frame.wall_time.to_numpy(dtype='datetime64[ns]'),
                seconds=(frame.wall_time-frame.wall_time.iloc[0]).dt.total_seconds().to_numpy(),
                cpu=frame.temperature_cpu_c.to_numpy(dtype=float),
                power=frame[[f'power_{r}_instantaneous_mw' for r in RAILS]].to_numpy(dtype=float),
                gpu=frame.gpu_utilization_pct.to_numpy(dtype=float),
                active=frame[[f'cpu{i}_active' for i in range(12)]].to_numpy(dtype=float),
                loads=frame[[f'cpu{i}_utilization_pct' for i in range(12)]].to_numpy(dtype=float),
                raw_index=frame.raw_sample_index.to_numpy())


def causal_features(data, origin_index, cfg):
    """Only slices ending at origin_index are read; no target information here."""
    time=data['seconds'];end=origin_index
    start_time=time[end]-cfg['history_s']
    if start_time<time[0]:return None,'history_before_window',None
    start=int(np.searchsorted(time,start_time))
    if time[start]!=start_time:return None,'history_start_missing',None
    sl=slice(start,end+1)
    offsets=time[sl]-time[end]
    intervals=np.diff(offsets)
    if np.any(intervals<=0) or np.any(intervals>cfg['max_history_interval_s']):
        return None,'history_gap',None
    temp=data['cpu'][sl];power=data['power'][sl];gpu=data['gpu'][sl]
    active=data['active'][sl];loads=data['loads'][sl]
    if (not all(np.isfinite(x).all() for x in (temp,power,gpu,active)) or
        not np.isin(active,[0,1]).all() or np.any(active.sum(axis=1)==0) or
        not np.isfinite(loads[active==1]).all()):
        return None,'nonfinite_features',None
    centered=offsets-offsets.mean()
    slope=float(centered@(temp-temp.mean())/(centered@centered))
    cpu_sample_mean=np.where(active==1,loads,0).sum(axis=1)/active.sum(axis=1)
    features=np.r_[temp[-1],temp.mean(),slope,temp.std(ddof=cfg['std_ddof']),
                   power.mean(axis=0),gpu.mean(),cpu_sample_mean.mean(),active[-1].sum()]
    if not np.isfinite(features).all():return None,'nonfinite_features',None
    return features,None,dict(start_index=start,end_index=end,n_history_samples=len(temp),
                              max_history_interval_s=float(intervals.max()))


def eligible_examples(window, data, cfg):
    rows=[];reports=[]
    time=data['seconds'];names=feature_names(cfg)
    cached=[causal_features(data,i,cfg) for i in range(len(time))]
    split='development' if window['recording_id'] in cfg['development_windows'] else 'heldout'
    for horizon in cfg['horizons_s']:
        rejected={r:0 for r in cfg['rejection_precedence']};eligible=0
        for i,t in enumerate(time):
            features,reason,history=cached[i]
            target_time=t+horizon
            target_index=int(np.searchsorted(time,target_time))
            if reason is None:
                if target_time>time[-1]:reason='target_after_window'
                elif target_index==len(time) or time[target_index]!=target_time:reason='target_timestamp_missing'
                elif not np.isfinite(data['cpu'][target_index]):reason='nonfinite_target'
            if reason:
                rejected[reason]+=1;continue
            row=dict(recording_id=window['recording_id'],window_id=window['window_id'],split=split,
                     group=window['proposed_validation_group'],horizon_s=horizon,
                     origin_timestamp=str(data['times'][i]),target_timestamp=str(data['times'][target_index]),
                     history_start_timestamp=str(data['times'][history['start_index']]),
                     origin_raw_index=int(data['raw_index'][i]),target_raw_index=int(data['raw_index'][target_index]),
                     history_start_raw_index=int(data['raw_index'][history['start_index']]),
                     n_history_samples=history['n_history_samples'],max_history_interval_s=history['max_history_interval_s'],
                     observed_c=float(data['cpu'][target_index]),delta_target_c=float(data['cpu'][target_index]-data['cpu'][i]))
            row.update(zip(names,features.tolist()));rows.append(row);eligible+=1
        reports.append(dict(recording_id=window['recording_id'],split=split,group=window['proposed_validation_group'],
                            horizon_s=horizon,potential_origins=len(time),eligible_origins=eligible,
                            rejected_origins=sum(rejected.values()),**rejected))
    return rows,reports


def window_weights(frame):
    counts=frame.groupby('recording_id').size().to_dict()
    weights=np.array([1/(len(counts)*counts[w]) for w in frame.recording_id],dtype=float)
    assert abs(weights.sum()-1)<1e-12
    return weights,counts


def fit_ridge(frame,names,cfg):
    if set(frame.split)!={'development'} or not set(frame.group).issubset(cfg['development_groups']):
        raise ValueError('Attempt to fit outside development groups')
    x=frame[names].to_numpy(dtype=float);y=frame.delta_target_c.to_numpy(dtype=float)
    weights,counts=window_weights(frame)
    mean=weights@x;variance=weights@((x-mean)**2);std=np.sqrt(variance)
    constant=std<=cfg['constant_std_threshold'];scale=np.where(constant,1.,std)
    z=(x-mean)/scale;z[:,constant]=0
    # Center explicitly so the intercept is unpenalized even under rounding.
    zmean=weights@z;ymean=float(weights@y)
    zc=z-zmean;yc=y-ymean
    beta=np.linalg.solve(zc.T@(weights[:,None]*zc)+cfg['ridge_lambda']*np.eye(len(names)),zc.T@(weights*yc))
    beta[constant]=0
    intercept=float(ymean-zmean@beta)
    return dict(features=names,feature_mean=mean.tolist(),feature_std=std.tolist(),feature_scale=scale.tolist(),
                constant_features=[n for n,c in zip(names,constant) if c],constant_mask=constant.tolist(),
                coefficients_standardized=beta.tolist(),intercept_delta_c=intercept,
                lambda_value=cfg['ridge_lambda'],intercept_penalized=False,
                objective=cfg['objective'],development_counts={k:int(v) for k,v in counts.items()},
                total_weight_by_window={k:1/len(counts) for k in counts},training_groups=sorted(set(frame.group)),
                training_n=len(frame),feature_min=x.min(axis=0).tolist(),feature_max=x.max(axis=0).tolist(),
                training_origin_sha256=hashlib.sha256(frame[['recording_id','origin_timestamp','target_timestamp']].to_csv(index=False).encode()).hexdigest())


def ridge_predict(model,x):
    x=np.asarray(x,dtype=float)
    z=(x-np.asarray(model['feature_mean']))/np.asarray(model['feature_scale'])
    z[...,np.asarray(model['constant_mask'],dtype=bool)]=0
    return model['intercept_delta_c']+z@np.asarray(model['coefficients_standardized'])


def predict_one(method,features,horizon,models):
    if method=='persistence':return float(features[0])
    if method=='local_trend':return float(features[0]+horizon*features[2])
    model=models[method]
    return float(features[0]+ridge_predict(model,features[:len(model['features'])]))


def metrics(observed,predicted):
    error=np.asarray(predicted)-np.asarray(observed)
    return dict(n=len(error),mae_c=float(np.abs(error).mean()),rmse_c=float(np.sqrt((error**2).mean())),
                signed_mean_error_c=float(error.mean()),p95_absolute_error_c=float(np.quantile(np.abs(error),.95)))


def equal_window_mae(per_window):
    return float(per_window.mae_c.mean())
