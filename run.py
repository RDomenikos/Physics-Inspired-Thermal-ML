"""Portable routes to existing Thermal ML code; saved replay is the default."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
import argparse,subprocess,sys,importlib.util,json

def load_cv():
    sys.path.insert(0,str(ROOT/'grouped/source'))
    spec=importlib.util.spec_from_file_location('grouped_runner',ROOT/'grouped/run.py')
    cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)
    cv.OUT=ROOT/'grouped/local/results';cv.OUT.mkdir(parents=True,exist_ok=True)
    return cv

def replay(name):
    import numpy as np
    import pandas as pd
    r=load_cv();device='jetson' if name=='jetson_inference' else 'intel'
    path=ROOT/'fixed'/device/'local/features.parquet'
    if not path.exists():raise SystemExit('Prepare '+device+' first: python -B run.py fixed '+device+' --source-dir PATH')
    frame=r.dataset(name,path);a=r.read(ROOT/'grouped/results'/(name+'_models.json'));pieces=[]
    for s in a['fold_states']:
        f=frame.loc[frame.outer_group.eq(s['fold'])];m=a['models'][s['receiver_key']];c=s['correction'];family=s['family']
        if name=='jetson_inference':
            if family=='Ridge':base=r.jr.ridge_delta(m,f);delta=s['alpha']*r.jr.controlled_delta(c,f)
            else:
                x=f[r.jn.FEATURES].to_numpy(float)
                base=r.jn.network_delta(r.jn.prepare_network(m),x)
                delta=s['alpha']*(c['c']+r.jn.physics_design(c,x)@c['gamma'])
        else:
            base=r.it.baseline(m,f);delta=s['alpha']*(c['c']+r.it.design(c,f)@c['gamma'])
        variants=[(family,np.zeros(len(f)),s['receiver_valid']),(family+'_physics',delta,s['correction_valid'])]
        if family=='H2':variants.append(('H2_bounded',s['bound_s_c']*np.tanh(delta/s['bound_s_c']),s['correction_valid']))
        for method,added,valid in variants:
            p=f[['origin_id','recording_id','origin_timestamp','target_timestamp','observed_c','delta_target_c']].copy()
            p['dataset']=name;p['fold']=s['fold'];p['method']=method;p['seed']=s['seed'];p['baseline_delta_c']=base;p['correction_c']=added;p['predicted_delta_c']=base+added;p['valid']=valid;p['bound_s_c']=s['bound_s_c'] if method=='H2_bounded' else np.nan
            pieces.append(p)
    pred=pd.concat(pieces,ignore_index=True)
    pred.to_parquet(r.OUT/(name+'_predictions.parquet'),index=False)
    r.verify(frame,pred,a) # Original scaling/group/residual checks; no coefficient fitting.
    r.summarize()
    fresh=pd.read_csv(r.OUT/'seed_metrics.csv');ref=pd.read_csv(ROOT/'grouped/results/seed_metrics.csv')
    keys=['dataset','fold','method','seed','recording_id']
    joined=fresh.loc[fresh.dataset.eq(name)].merge(ref.loc[ref.dataset.eq(name)],on=keys,suffixes=('_new','_ref'),validate='one_to_one')
    assert len(joined)==len(ref.loc[ref.dataset.eq(name)])
    for col in r.METRICS:np.testing.assert_allclose(joined[col+'_new'],joined[col+'_ref'],atol=1e-9,rtol=0,equal_nan=True)
    print(name,': saved-state replay and frozen seed/per-recording references verified',flush=True)
    return r,frame,a,pred

def refinement():
    import numpy as np
    import pandas as pd
    r,frame,a,pred=replay('jetson_inference')
    out=ROOT/'refinement/local';out.mkdir(parents=True,exist_ok=True)
    parts=[];checks=[]
    for s in a['fold_states']:
        if s['family']!='Ridge':continue
        assert s['alpha']==1
        f=frame.loc[frame.outer_group.eq(s['fold'])];c=s['correction']
        base=r.jr.ridge_delta(a['models'][s['receiver_key']],f)
        _,phi=r.jr.centered_design(c,f)
        original=s['alpha']*r.jr.controlled_delta(c,f)
        removed=s['alpha']*(phi@np.asarray(c['gamma']))
        np.testing.assert_allclose(removed,original-s['alpha']*c['intercept_delta_c'],atol=1e-12,rtol=0)
        saved=pred.loc[pred.fold.eq(s['fold'])&pred.method.eq('Ridge')]
        np.testing.assert_array_equal(base,saved.predicted_delta_c)
        for method,added in [('baseline',np.zeros(len(f))),('original',original),('removed',removed)]:
            p=f[['origin_id','recording_id','origin_timestamp','target_timestamp','observed_c','delta_target_c']].copy()
            p['fold']=s['fold'];p['method']=method;p['predicted_delta_c']=base+added;p['error_c']=base+added-f.delta_target_c.to_numpy();p['correction_c']=added
            p['fold_weight']=r.weights(f);parts.append(p)
        checks.append(dict(fold=s['fold'],alpha=s['alpha'],intercept_c=c['intercept_delta_c'],identity_max_abs_c=float(np.max(abs((base+removed)-(base+original-s['alpha']*c['intercept_delta_c']))))))
    p=pd.concat(parts,ignore_index=True);p.to_parquet(out/'predictions.parquet',index=False);rows=[]
    for (fold,method),g in p.groupby(['fold','method']):
        for rid in ['ALL']+sorted(g.recording_id.unique()):
            v=g if rid=='ALL' else g.loc[g.recording_id.eq(rid)];w=r.weights(v);e=v.error_c.to_numpy()
            row=dict(level='fold' if rid=='ALL' else 'recording',fold=fold,recording_id=rid,method=method,n=len(v),mae_c=float(w@abs(e)),rmse_c=float(np.sqrt(w@e**2)),bias_c=float(w@e),mean_correction_c=float(w@v.correction_c))
            for ref in ['baseline','original']:
                b=p.loc[p.fold.eq(fold)&p.method.eq(ref)].set_index('origin_id').loc[v.origin_id];be=b.error_c.to_numpy();change=abs(e)-abs(be)
                row['mae_c_improvement_vs_'+ref]=float(w@(abs(be)-abs(e)))
                row['rmse_c_improvement_vs_'+ref]=float(np.sqrt(w@be**2)-np.sqrt(w@e**2))
                for label,mask in [('improved',change < -1e-10),('unchanged',abs(change)<=1e-10),('worsened',change>1e-10)]:row[label+'_fraction_vs_'+ref]=float(w@mask)
            rows.append(row)
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    frozen=pd.read_csv(ROOT/'refinement/reference/metrics.csv');keys=['level','fold','recording_id','method']
    j=metrics.merge(frozen.loc[frozen.level.isin(['fold','recording'])],on=keys,suffixes=('_new','_ref'),validate='one_to_one');assert len(j)==len(metrics)
    for col in ['mae_c','rmse_c','bias_c','mean_correction_c']:np.testing.assert_allclose(j[col+'_new'],j[col+'_ref'],atol=1e-10,rtol=0)
    summaries=[];paired=[]
    for method,g in metrics.loc[metrics.level.eq('fold')].groupby('method'):
        row=dict(method=method,folds=len(g))
        for col in ['mae_c','rmse_c','bias_c','mean_correction_c']:row[col+'_mean']=g[col].mean();row[col+'_sd']=g[col].std(ddof=1)
        summaries.append(row)
        for ref in ['baseline','original']:
            z=dict(method=method,reference=ref)
            for metric in ['mae_c','rmse_c']:
                d=g[metric+'_improvement_vs_'+ref];z[metric+'_improvement_mean']=d.mean();z[metric+'_improvement_sd']=d.std(ddof=1)
                denominator=metrics.loc[metrics.level.eq('fold')&metrics.method.eq(ref),metric].mean()
                z[metric+'_mean_error_reduction_pct']=100*d.mean()/denominator
                for label,mask in [('wins',d>1e-10),('ties',abs(d)<=1e-10),('losses',d< -1e-10)]:z[metric+'_'+label]=int(mask.sum())
            paired.append(z)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_summary.csv',index=False)
    r.save(out/'verification.json',dict(label='Retrospective intercept-removal refinement, evaluated on all three existing grouped folds without refitting.',generated_from='saved states and prepared evaluation inputs',new_fits=0,frozen_reference_rows_verified=len(j),folds=checks))
    print('Refinement: saved coordinates/slopes/alpha and independent frozen references verified')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='route',required=True)
    f=sub.add_parser('fixed');f.add_argument('device',choices=['jetson','intel']);f.add_argument('--source-dir',type=Path);f.add_argument('--refit-check',action='store_true')
    g=sub.add_parser('grouped');g.add_argument('dataset',choices=['jetson_inference','intel','all']);g.add_argument('--refit',action='store_true');g.add_argument('--output-dir',type=Path)
    sub.add_parser('refinement');sub.add_parser('report')
    a=p.parse_args()
    if a.route=='fixed':
        cmd=[sys.executable,'-B',str(ROOT/'fixed/run.py'),a.device]
        if a.source_dir:cmd+=['--source-dir',str(a.source_dir.resolve())]
        if a.refit_check:cmd+=['--refit-check']
        subprocess.run(cmd,check=True)
    elif a.route=='grouped':
        for name in (['jetson_inference','intel'] if a.dataset=='all' else [a.dataset]):
            if a.refit:
                if not a.output_dir:p.error('--refit requires a new --output-dir; released states are never overwritten')
                device='jetson' if name=='jetson_inference' else 'intel'
                subprocess.run([sys.executable,'-B',str(ROOT/'grouped/run.py'),'--dataset',name,'--features',str(ROOT/'fixed'/device/'local/features.parquet'),'--output-dir',str(a.output_dir.resolve())],check=True)
            else:replay(name)
    elif a.route=='refinement':refinement()
    else:subprocess.run([sys.executable,'-B',str(ROOT/'report.py')],check=True)
if __name__=='__main__':main()
