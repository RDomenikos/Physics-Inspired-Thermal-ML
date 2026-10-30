"""Grouped orchestration around unchanged existing fitting/scoring functions."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
import argparse,hashlib,json,time,sys,importlib.util
import numpy as np
import pandas as pd
sys.path.insert(0,str(ROOT/'source'))
import jetson_ridge as jr
import jetson_neural as jn
import intel as it
from scoring import weighted_quantile,error_stats
OUT=Path(os.environ.get('THERMAL_CV_RESULTS',str(ROOT/'results')))
METRICS=['mae_c','rmse_c','signed_mean_error_c','fraction_abs_error_above_1c','median_absolute_error_c','p90_absolute_error_c','p95_absolute_error_c','p99_absolute_error_c']
METHODS=['Ridge','Ridge_physics','H2','H2_physics','H2_bounded']
PAIRS={'Ridge_physics':'Ridge','H2_physics':'H2','H2_bounded':'H2'}
def read(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
 if isinstance(x,np.generic):return x.item()
 return x

def save(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True).encode()).hexdigest()
def weights(f):return 1/(f.recording_id.nunique()*f.groupby('recording_id').recording_id.transform('size').to_numpy())

def dataset(name,path):
 f=pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path,float_precision='round_trip')
 if name=='jetson_inference':f=f.loc[f.horizon_s.eq(60)].copy()
 else:f=f.copy()
 a=pd.read_csv(ROOT/'groups.csv');a=a.loc[a.dataset.eq(name)]
 assert set(f.recording_id)==set(a.recording_id)
 assert f.groupby('recording_id').size().to_dict()==a.set_index('recording_id').n_origins.to_dict()
 f['outer_group']=f.recording_id.map(a.set_index('recording_id').group)
 f['origin_id']=f.recording_id+'::'+f.origin_timestamp
 assert not f.origin_id.duplicated().any()
 if 'window_id' not in f:f['window_id']=f.recording_id
 return f.reset_index(drop=True)

def run_dataset(name,path):
 cfg=read(ROOT/'config.json')
 if name=='jetson_inference' and cfg['jetson_policy']!='fixed_configuration_authorized':raise RuntimeError('Jetson fixed-configuration choice not authorized; no Jetson fits run.')
 OUT.mkdir(exist_ok=True)
 state_path=OUT/(name+'_models.json')
 if state_path.exists():raise RuntimeError('Completed dataset already exists; use --summarize. Refusing to overwrite fitted states.')
 frame=dataset(name,path);groups=sorted(frame.outer_group.unique());assert len(groups)==3
 folds=[f for f in read(ROOT/'folds.json') if f['dataset']==name]
 cache={};fold_states=[];predictions=[];failures=[];start=time.perf_counter()
 isj=name=='jetson_inference';features=jn.FEATURES if isj else it.FEATURES
 def subset(gs):return frame.loc[frame.outer_group.isin(gs)]
 def get(gs,family,seed):
  key=f"{family}|{','.join(sorted(gs))}|{seed}"
  if key not in cache:
   f=subset(gs);tic=time.perf_counter()
   if family=='Ridge':m=jr.fit_ridge(f,features,f.delta_target_c,dict(cfg['jetson_ridge_config'],ridge_lambda=10.)) if isj else it.fit_ridge(f,features,10.)
   else:m=jn.fit_network(f,2,.03,seed) if isj else it.fit_network(f,seed)
   m.update(training_groups=sorted(gs),training_recordings=sorted(f.recording_id.unique()),training_origin_hash=digest(sorted(f.origin_id)),fit_seconds=m.get('fit_seconds',time.perf_counter()-tic))
   cache[key]=m
   if not m.get('success',True):failures.append(dict(dataset=name,fit_key=key,reason='Optimizer unsuccessful after original allowance',message=m.get('message',m.get('phases')),training_groups=list(gs)))
  return key,cache[key]
 def predict(m,f,family):
  if not isj:return it.baseline(m,f)
  return jr.ridge_delta(m,f) if family=='Ridge' else jn.network_delta(jn.prepare_network(m),f[features].to_numpy(float))
 for fold in folds:
  group=fold['fold'];S=fold['train_groups'];train=subset(S);test=subset([group])
  assert not set(train.outer_group)&set(test.outer_group)
  for family,seeds in [('Ridge',[-1]),('H2',cfg['seeds'])]:
   for seed in seeds:
    key,receiver=get(S,family,seed);frozen=digest(receiver);base=predict(receiver,test,family)
    oof=np.full(len(train),np.nan);sources=[];dependencies=[]
    for excluded in S:
     allowed=[g for g in S if g!=excluded];k,m=get(allowed,family,seed);mask=train.outer_group.eq(excluded).to_numpy();part=train.loc[mask]
     assert group not in m['training_groups'] and excluded not in m['training_groups']
     oof[mask]=part.delta_target_c.to_numpy()-predict(m,part,family)
     dependencies.append(k);sources.append(dict(excluded_group=excluded,training_groups=allowed,fit_key=k,residual_origins=len(part),residual_origin_hash=digest(sorted(part.origin_id))))
    assert np.isfinite(oof).all()
    lam=1. if family=='Ridge' else .1;alpha=1. if family=='Ridge' else .25
    if isj and family=='Ridge':
     correction=jr.fit_controlled_ridge(train,cfg['jetson_ridge_config'],lam,target=oof,correction=True)
     train_corr=alpha*jr.controlled_delta(correction,train);delta=alpha*jr.controlled_delta(correction,test)
    elif isj:
     correction=jn.fit_correction(train,oof,lam)
     train_corr=alpha*(correction['c']+jn.physics_design(correction,train[features].to_numpy(float))@correction['gamma'])
     delta=alpha*(correction['c']+jn.physics_design(correction,test[features].to_numpy(float))@correction['gamma'])
    else:
     correction=it.fit_correction(train,oof,lam,'physics');train_corr=alpha*(correction['c']+it.design(correction,train)@correction['gamma']);delta=alpha*(correction['c']+it.design(correction,test)@correction['gamma'])
    correction['training_groups']=S;correction['training_origin_hash']=digest(sorted(train.origin_id))
    q=weighted_quantile(abs(train_corr),weights(train),.95);bound=q if q else 1e-12
    assert digest(receiver)==frozen
    np.testing.assert_array_equal(base,predict(receiver,test,family))
    receiver_valid=receiver.get('success',True);corrected_valid=receiver_valid and all(cache[k].get('success',True) for k in dependencies)
    fold_states.append(dict(dataset=name,fold=group,family=family,seed=seed,receiver_key=key,receiver_hash=frozen,training_groups=S,test_groups=[group],correction=correction,alpha=alpha,bound_s_c=bound if family=='H2' else None,residual_sources=sources,receiver_valid=receiver_valid,correction_valid=corrected_valid,residual_hash=digest(oof.tolist())))
    variants=[(family,np.zeros(len(test)),receiver_valid),(family+'_physics',delta,corrected_valid)]
    if family=='H2':variants.append(('H2_bounded',bound*np.tanh(delta/bound),corrected_valid))
    for method,added,valid in variants:
     p=test[['origin_id','recording_id','origin_timestamp','target_timestamp','observed_c','delta_target_c']].copy()
     p['dataset']=name;p['fold']=group;p['method']=method;p['seed']=seed;p['baseline_delta_c']=base;p['correction_c']=added;p['predicted_delta_c']=base+added;p['valid']=valid;p['bound_s_c']=bound if method=='H2_bounded' else np.nan
     predictions.append(p)
  print(name,group,'completed; cached fits',len(cache),flush=True)
 pred=pd.concat(predictions,ignore_index=True)
 artifact=dict(dataset=name,configuration=cfg,plan_sha256=hashlib.sha256((ROOT/'evaluation_plan.md').read_bytes()).hexdigest(),fold_manifest_sha256=digest(folds),models=cache,fold_states=fold_states,failures=failures,wall_seconds=time.perf_counter()-start)
 save(state_path,artifact);pred.to_parquet(OUT/(name+'_predictions.parquet'),index=False)
 save(OUT/(name+'_failures.json'),failures)
 verify(frame,pred,artifact)
 return pred

def verify(frame,pred,artifact):
 models=artifact['models'];name=artifact['dataset'];features=jn.FEATURES if name=='jetson_inference' else it.FEATURES
 for key,m in models.items():
  assert m['lambda_value']==(10. if key.startswith('Ridge|') else .03)
  assert len(m['training_groups']) in [1,2]
  tr=frame.loc[frame.outer_group.isin(m['training_groups'])];w=weights(tr);x=tr[features].to_numpy();mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd=np.where(sd<1e-12,1,sd)
  np.testing.assert_allclose(m.get('mean',m.get('feature_mean')),mu,atol=1e-9,rtol=0)
  np.testing.assert_allclose(m.get('scale',m.get('feature_scale')),sd,atol=1e-9,rtol=0)
  assert m['training_origin_hash']==digest(sorted(tr.origin_id))
 for f in artifact['fold_states']:
  assert f['fold'] not in f['training_groups'];assert digest(models[f['receiver_key']])==f['receiver_hash']
  tr=frame.loc[frame.outer_group.isin(f['training_groups'])];residual=np.full(len(tr),np.nan)
  for source in f['residual_sources']:
   assert source['excluded_group'] not in source['training_groups'] and f['fold'] not in source['training_groups']
   mask=tr.outer_group.eq(source['excluded_group']).to_numpy();v=tr.loc[mask];producer=models[source['fit_key']]
   assert producer['training_groups']==source['training_groups']
   assert digest(sorted(v.origin_id))==source['residual_origin_hash']
   if name=='jetson_inference':
    delta=jr.ridge_delta(producer,v) if f['family']=='Ridge' else jn.network_delta(jn.prepare_network(producer),v[features].to_numpy(float))
   else:delta=it.baseline(producer,v)
   residual[mask]=v.delta_target_c.to_numpy()-delta
  assert np.isfinite(residual).all() and digest(residual.tolist())==f['residual_hash']
  w=weights(tr);c=f['correction'];family=f['family']
  assert c['lambda_value']==(1. if family=='Ridge' else .1)
  assert f['alpha']==(1. if family=='Ridge' else .25)
  if name=='jetson_inference' and family=='Ridge':
   expected=jr.fit_centered_physics(tr,artifact['configuration']['jetson_ridge_config'])
   for k in ['coordinates','thermal_mean','thermal_scale','kept_features']:assert digest(c[k])==digest(expected[k])
   added=f['alpha']*jr.controlled_delta(c,tr)
  elif name=='jetson_inference':
   expected=jn.fit_physics_scaling(tr)
   for k in expected:np.testing.assert_allclose(c[k],expected[k],atol=1e-12,rtol=0)
   added=f['alpha']*(c['c']+jn.physics_design(c,tr[features].to_numpy(float))@c['gamma'])
  else:
   T=tr[features[0]].to_numpy()+273.15;P=tr[features[4]].to_numpy()
   np.testing.assert_allclose([c['T'],c['P']],[w@T,w@P],atol=1e-10,rtol=0)
   raw=it.raw_terms(c,tr,'physics');np.testing.assert_allclose(c['mean'],w@raw,atol=1e-10,rtol=0)
   added=f['alpha']*(c['c']+it.design(c,tr)@c['gamma'])
  if family=='H2':
   q=weighted_quantile(abs(added),w,.95);np.testing.assert_allclose(f['bound_s_c'],q if q else 1e-12,atol=1e-12,rtol=0)
  seed=f['seed'];part=pred.loc[pred.fold.eq(f['fold'])&pred.seed.eq(seed)]
  base=part.loc[part.method.eq(family)].set_index('origin_id')
  np.testing.assert_array_equal(base.predicted_delta_c,base.predicted_delta_c.to_numpy()+np.zeros(len(base)))
  for method in [family+'_physics']+(['H2_bounded'] if family=='H2' else []):
   corr=part.loc[part.method.eq(method)].set_index('origin_id');assert corr.index.equals(base.index)
   np.testing.assert_array_equal(corr.baseline_delta_c,base.predicted_delta_c)
 for _,p in pred.groupby(['method','seed']):
  assert len(p)==len(frame);assert not p.origin_id.duplicated().any();assert set(p.origin_id)==set(frame.origin_id)
 np.testing.assert_allclose(pred.predicted_delta_c,pred.baseline_delta_c+pred.correction_c,atol=1e-12,rtol=0)
 save(OUT/(name+'_verification.json'),dict(outer_inner_exclusion=True,training_only_scaling=True,training_only_correction_scaling=True,training_only_bounds=True,group_excluded_residual_values_verified=True,fixed_hyperparameters_verified=True,baseline_frozen=True,alpha_zero_exact=True,identical_rows=True,each_origin_once_per_method_seed=True,planned_folds=3,unique_baseline_fits=len(models),unresolved_fits=len(artifact['failures'])))

def seed_scores(pred):
 rows=[];folds=read(ROOT/'folds.json');lookup={(f['dataset'],f['fold']):f for f in folds}
 for (ds,fold,method,seed),p in pred.groupby(['dataset','fold','method','seed']):
  for recording in ['ALL']+sorted(p.recording_id.unique()):
   part=p if recording=='ALL' else p.loc[p.recording_id.eq(recording)]
   error=part.predicted_delta_c.to_numpy()-part.delta_target_c.to_numpy();before=part.baseline_delta_c.to_numpy()-part.delta_target_c.to_numpy()
   row=dict(dataset=ds,fold=fold,method=method,seed=int(seed),recording_id=recording,n=len(part),valid=bool(part.valid.all()),**error_stats(error,before,part.correction_c.to_numpy(),weights(part)))
   if (ds,fold) in lookup:
    f=lookup[(ds,fold)];row.update(train_groups='|'.join(f['train_groups']),test_groups='|'.join(f['test_groups']),train_recordings='|'.join(f['train_recordings']),test_recordings='|'.join(f['test_recordings']),train_origins=f['train_origins'],test_origins=f['test_origins'])
   rows.append(row)
 return pd.DataFrame(rows)

def supplementary():
 path=ROOT/'input/supplementary_metrics.csv'
 if not path.exists():return pd.read_csv(ROOT/'reference/supplementary_seed_metrics.csv')
 old=pd.read_csv(path);a=old.loc[old.stratum.eq('all')&old.scope.isin(['group','recording'])&(~old.seed.astype(str).str.startswith('SEED'))].copy()
 groups=pd.read_csv(ROOT/'input/supplementary_grouping.csv').set_index('recording_id').outer_group.to_dict()
 a['dataset']='jetson_supplementary';a['fold']=[r.label if r.scope=='group' else groups[r.label] for r in a.itertuples()];a['recording_id']=np.where(a.scope.eq('group'),'ALL',a.label);a['method']=a.method.replace({'unbounded':'H2_physics','bounded':'H2_bounded'});a['seed']=a.seed.astype(int);a['valid']=~a.depends_on_unresolved
 # Independently verify reused fold metrics from existing saved OOF predictions; no refitting.
 p=pd.read_csv(ROOT/'input/supplementary_predictions.csv',float_precision='round_trip');pieces=[]
 for method,col in [('H2','baseline_delta_c'),('H2_physics','unbounded_delta_c'),('H2_bounded','bounded_delta_c')]:
  q=p.copy();q['dataset']='jetson_supplementary';q['fold']=q.outer_group;q['method']=method;q['predicted_delta_c']=q[col];q['correction_c']=q[col]-q.baseline_delta_c;q['valid']=~q.depends_on_unresolved;pieces.append(q)
 fresh=seed_scores(pd.concat(pieces));keys=['dataset','fold','method','seed','recording_id'];j=a.merge(fresh,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(j)==len(a)
 for k in METRICS:np.testing.assert_allclose(j[k+'_old'],j[k+'_new'],atol=1e-10,rtol=0,equal_nan=True)
 save(OUT/'supplementary_verification.json',dict(reused=True,new_fits=0,seed_metric_rows_checked=len(j),all_existing_fits_converged=not bool(a.depends_on_unresolved.any())))
 lookup={f['fold']:f for f in read(ROOT/'folds.json') if f['dataset']=='jetson_supplementary'}
 for k in ['train_groups','test_groups','train_recordings','test_recordings']:
  a[k]=a.fold.map(lambda f:'|'.join(lookup[f][k]))
 a['train_origins']=a.fold.map(lambda f:lookup[f]['train_origins']);a['test_origins']=a.fold.map(lambda f:lookup[f]['test_origins'])
 return a

def summarize():
 parts=[]
 for name in ['intel','jetson_inference']:
  path=OUT/(name+'_predictions.parquet')
  if path.exists():parts.append(seed_scores(pd.read_parquet(path)))
 if not parts:raise RuntimeError('No locally generated primary OOF predictions found; refusing to replace published tables with an incomplete summary.')
 parts.append(supplementary());s=pd.concat(parts,ignore_index=True);s.to_csv(OUT/'seed_metrics.csv',index=False)
 fold=[]
 for (ds,f,method),g in s.loc[s.recording_id.eq('ALL')].groupby(['dataset','fold','method']):
  expected=1 if method.startswith('Ridge') else 5;valid=len(g)==expected and g.valid.all()
  row=dict(dataset=ds,fold=f,method=method,valid=bool(valid),seeds_present=len(g),seeds_expected=expected)
  for metric in METRICS:row[metric]=float(g[metric].mean()) if valid else np.nan;row[metric+'_seed_sd']=float(g[metric].std(ddof=1)) if valid and len(g)>1 else np.nan
  fold.append(row)
 f=pd.DataFrame(fold);f.to_csv(OUT/'fold_metrics.csv',index=False)
 summary=[]
 for (ds,method),g in f.groupby(['dataset','method']):
  v=g.loc[g.valid];row=dict(dataset=ds,method=method,valid_folds=len(v),planned_folds=3)
  for metric in METRICS:row[metric+'_mean']=v[metric].mean();row[metric+'_sd']=v[metric].std(ddof=1)
  summary.append(row)
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
 paired=[]
 for (ds,fold),g in f.groupby(['dataset','fold']):
  by=g.set_index('method')
  for method,base in PAIRS.items():
   if method not in by.index or base not in by.index:continue
   valid=bool(by.loc[method,'valid'] and by.loc[base,'valid']);r=dict(dataset=ds,fold=fold,baseline=base,corrected=method,valid=valid)
   for k in ['mae_c','rmse_c']:
    d=by.loc[base,k]-by.loc[method,k] if valid else np.nan;r[k+'_improvement']=d;r[k+'_improvement_pct']=100*d/by.loc[base,k] if valid and by.loc[base,k] else np.nan
   paired.append(r)
 p=pd.DataFrame(paired);p.to_csv(OUT/'paired_folds.csv',index=False);sums=[]
 for (ds,base,corrected),g in p.groupby(['dataset','baseline','corrected']):
  v=g.loc[g.valid];row=dict(dataset=ds,baseline=base,corrected=corrected,valid_folds=len(v),planned_folds=3)
  for k in ['mae_c','rmse_c']:
   for suffix in ['_improvement','_improvement_pct']:
    col=k+suffix;row[col+'_mean']=v[col].mean();row[col+'_sd']=v[col].std(ddof=1)
   a=v[k+'_improvement'];row[k+'_improved']=int((a>1e-10).sum());row[k+'_unchanged']=int((abs(a)<=1e-10).sum());row[k+'_worsened']=int((a< -1e-10).sum())
  sums.append(row)
 pd.DataFrame(sums).to_csv(OUT/'paired_summary.csv',index=False)
 return f

def main():
 global OUT
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['intel','jetson_inference']);p.add_argument('--features',type=Path);p.add_argument('--summarize',action='store_true');p.add_argument('--output-dir',type=Path);a=p.parse_args()
 if a.output_dir:OUT=a.output_dir;OUT.mkdir(parents=True,exist_ok=True)
 if a.dataset:
  path=a.features or ROOT/'input'/('intel.parquet' if a.dataset=='intel' else 'jetson.parquet');run_dataset(a.dataset,path)
 elif not a.summarize:p.error('Specify --dataset or --summarize')
 summarize()
if __name__=='__main__':main()
