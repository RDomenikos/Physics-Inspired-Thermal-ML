"""Portable saved-model evidence replay. No fitting or timing benchmarks."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import argparse,json,hashlib
import numpy as np
import pandas as pd

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def raw_terms(raw,coord,generic=False):
 theta=raw[:,0]+273.15-coord['temperature_reference_k'];s=coord['temperature_scale_k'];x=theta/s
 p=(raw[:,5]/1000-coord['power_mean_w'])/coord['power_scale_w'];T=coord['temperature_reference_k']
 if generic:return np.column_stack((x*x,x*x*x,p*x))
 return np.column_stack((x*np.abs(x)**.25,(6*T*T*theta*theta+4*T*theta**3+theta**4)/(4*T**3*s),p*x))
def design(model,raw,kind):
 if kind=='root':
  phi=(raw_terms(raw,model['coordinates'],model['family']=='generic_enriched')-model['thermal_mean'])/model['thermal_scale'];kept=model['kept_features']
 else:phi=(raw_terms(raw,model)-model['mean'])/model['scale'];kept=model['kept']
 phi[:,~np.asarray(kept,dtype=bool)]=0
 return phi

def predict(spec,frame,names):
 raw=frame[names].to_numpy(float);model=spec['baseline'];terms=np.zeros((len(frame),3));intercept=0.
 if spec['kind']=='persistence':base=np.zeros(len(frame))
 elif spec['kind']=='network':base=np.tanh(((raw-model['mean'])/model['scale'])@np.asarray(model['W'])+model['a'])@np.asarray(model['v'])+model['b']
 else:
  x=frame[model['features']].to_numpy(float);z=(x-model['feature_mean'])/model['feature_scale']
  base=model['intercept_delta_c']+z@np.asarray(model['coefficients_standardized'])
  if spec['kind']=='joint':base+=design(model,raw,'root')@np.asarray(model['gamma'])
 original=np.zeros(len(frame));corr=spec.get('correction')
 if corr:
  kind=spec['correction_kind'];terms=spec['alpha']*design(corr,raw,kind)*np.asarray(corr['gamma']);intercept=spec['alpha']*corr['intercept_delta_c' if kind=='root' else 'c'];original=terms.sum(axis=1)+intercept
 applied=original if spec.get('bound') is None else spec['bound']*np.tanh(original/spec['bound'])
 return frame.cpu_current_c.to_numpy()+base+applied,frame.cpu_current_c.to_numpy()+base,original,applied,terms,intercept

def weights(frame,group_balanced=False):
 n=frame.groupby('recording_id').recording_id.transform('size').to_numpy();w=1/(frame.recording_id.nunique()*n)
 if group_balanced:
  counts=frame[['group_id','recording_id']].drop_duplicates().groupby('group_id').size()
  w=1/(frame.group_id.nunique()*frame.group_id.map(counts).to_numpy()*n)
 return w

def quantile(x,w,q):
 i=np.argsort(x,kind='stable');return float(x[i[min(np.searchsorted(np.cumsum(w[i]/w.sum()),q),len(i)-1)]])
def stats(e,w):
 w=w/w.sum();a=np.abs(e)
 out=dict(mae_c=float(w@a),rmse_c=float(np.sqrt(w@(e*e))),signed_mean_error_c=float(w@e),fraction_above_1c=float(w@(a>1)))
 for k,q in [('median',.5),('p90',.9),('p95',.95),('p99',.99)]:out[k+'_absolute_error_c']=quantile(a,w,q)
 return out

def reproduce(out):
 cfg=read(ROOT/'config/study.json');reg=read(ROOT/'models/registry.json');ref=pd.read_parquet(ROOT/'results/reference/predictions.parquet');datasets={k:pd.read_parquet(ROOT/'data'/f'{k}.parquet').set_index('origin_id') for k in ref.dataset_id.unique()}
 pieces=[];maximum=0.;details=[]
 for mid,part in ref.groupby('model_id',sort=False):
  spec=reg[mid];frame=datasets[part.dataset_id.iloc[0]].loc[part.origin_id].reset_index();p,b,o,a,t,c=predict(spec,frame,cfg['features'])
  diff=float(np.max(np.abs(p-part.predicted_temperature_c.to_numpy())));maximum=max(maximum,diff)
  if diff>cfg['prediction_tolerance_c']:raise AssertionError(f'Prediction mismatch {mid}: {diff}')
  q=part.copy();q['predicted_temperature_c']=p;q['baseline_prediction_c']=b;q['original_physics_correction_c']=o;q['applied_physics_correction_c']=a
  q['correction_intercept_c']=c
  for j,key in enumerate(['C','R','I']):q['original_'+key+'_contribution_c']=t[:,j]
  q['signed_error_c']=p-q.observed_c.to_numpy();q['absolute_error_c']=np.abs(q.signed_error_c);q['squared_error_c']=q.signed_error_c**2
  lo,hi=spec['temperature_range'];q['inside_training_temperature_range']=q.cpu_current_c.between(lo,hi);q['training_min_c']=lo;q['training_max_c']=hi
  pieces.append(q)
  params=0 if spec['kind']=='persistence' else (13*model_width(spec)+1 if spec['kind']=='network' else len(spec['baseline']['coefficients_standardized'])+1)
  if spec['kind']=='joint':params+=3
  if spec.get('correction'):params+=4
  details.append(dict(model_id=mid,kind=spec['kind'],trainable_prediction_parameters=params,alpha=spec['alpha'],bound_s_c=spec.get('bound'),baseline_lambda=spec.get('baseline',{}).get('lambda_value') if spec.get('baseline') else None,correction_lambda=spec.get('correction',{}).get('lambda_value') if spec.get('correction') else None,source_experiment=spec['source_experiment']))
 pred=pd.concat(pieces).sort_index();assert len(pred)==len(ref)
 pred.to_parquet(out/'predictions.parquet',index=False);pred.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
 pd.DataFrame(details).to_csv(out/'model_details.csv',index=False)
 save(out/'verification.json',dict(reference_rows=len(ref),reproduced_rows=len(pred),maximum_prediction_difference_c=maximum,tolerance_c=cfg['prediction_tolerance_c'],new_fits=0,new_timing_measurements=0))
 print('Reproduced',len(pred),'predictions; max difference',maximum,flush=True)
 return pred

def model_width(spec):return len(spec['baseline']['a'])

def metrics(pred,out):
 rows=[]
 for (exp,run,split,seed,method),df in pred.groupby(['experiment_id','run_id','split','seed','method'],sort=False):
  scopes=[('recording',r,df.recording_id.eq(r).to_numpy()) for r in sorted(df.recording_id.unique())]
  scopes += [('group',g,df.group_id.eq(g).to_numpy()) for g in sorted(df.group_id.unique())]
  scopes += [('workload',g,df.workload_type.eq(g).to_numpy()) for g in sorted(df.workload_type.unique())]
  scopes += [('aggregate','equal_recording',np.ones(len(df),bool))]
  if exp=='grouped':scopes += [('aggregate','equal_group',np.ones(len(df),bool))]
  e=df.signed_error_c.to_numpy();base=df.baseline_prediction_c.to_numpy()-df.observed_c.to_numpy();correction=df.applied_physics_correction_c.to_numpy();inside=df.inside_training_temperature_range.to_numpy()
  for scope,label,mask in scopes:
   w=np.zeros(len(df));w[mask]=weights(df.loc[mask],label=='equal_group')
   for stratum,s in [('all',mask),('inside',mask&inside),('outside',mask&~inside)]:
    mass=w[s].sum();v=w[s]/mass if mass else w[s]
    row=dict(experiment_id=exp,run_id=run,split=split,seed=int(seed),method=method,comparator=df.comparator.iloc[0],scope=scope,label=label,stratum=stratum,n=int(s.sum()),weight_mass=float(mass),outside_fraction=float(w@(~inside)))
    if mass:
     row.update(stats(e[s],w[s]));row.update(correction_mean_c=float(v@correction[s]),correction_abs_mean_c=float(v@np.abs(correction[s])),wrong_direction_fraction=float(v@((base[s]*correction[s])>0)),harmful_overshoot_fraction=float(v@(((base[s]*e[s])<0)&(np.abs(e[s])>np.abs(base[s])))))
    rows.append(row)
 m=pd.DataFrame(rows);keys=['experiment_id','run_id','split','method','comparator','scope','label','stratum'];numeric=[c for c in m.select_dtypes(include=np.number) if c not in ['seed','n']]
 means=m.groupby(keys)[numeric].mean().reset_index();means['seed']='SEED_MEAN'
 sd=m.groupby(keys)[numeric].std(ddof=1).reset_index();sd['seed']='SEED_SD';m=pd.concat([m,means,sd],ignore_index=True)
 lookup=m.set_index(['experiment_id','split','seed','method','scope','label','stratum'])
 for metric in ['mae_c','rmse_c','fraction_above_1c']:
  changes=[];perc=[]
  for _,r in m.iterrows():
   key=(r.experiment_id,r.split,r.seed,r.comparator,r.scope,r.label,r.stratum)
   base=lookup.loc[key,metric] if key in lookup.index and r.seed!='SEED_SD' else np.nan
   changes.append(r[metric]-base);perc.append(100*(r[metric]-base)/base if base else np.nan)
  m[metric+'_minus_comparator']=changes;m[metric+'_pct_change']=perc
 m.to_csv(out/'metrics.csv',index=False)
 wins=[]
 for key,df in m.loc[m.seed.eq('SEED_MEAN')&m.stratum.eq('all')&m.scope.isin(['recording','group'])].groupby(['experiment_id','split','method','comparator','scope']):
  wins.append(dict(zip(['experiment_id','split','method','comparator','scope'],key),mae_wins=int((df.mae_c_minus_comparator<0).sum()),rmse_wins=int((df.rmse_c_minus_comparator<0).sum()),both_wins=int(((df.mae_c_minus_comparator<0)&(df.rmse_c_minus_comparator<0)).sum()),denominator=len(df)))
 pd.DataFrame(wins).to_csv(out/'win_counts.csv',index=False)
 # Independent groupby computation of headline balanced RMSE.
 for key,df in pred.groupby(['experiment_id','split','seed','method']):
  per=df.groupby('recording_id')[['absolute_error_c','squared_error_c']].mean().mean()
  row=m.loc[m.experiment_id.eq(key[0])&m.split.eq(key[1])&m.seed.eq(key[2])&m.method.eq(key[3])&m.scope.eq('aggregate')&m.label.eq('equal_recording')&m.stratum.eq('all')].iloc[0]
  np.testing.assert_allclose([row.mae_c,row.rmse_c],[per.absolute_error_c,np.sqrt(per.squared_error_c)],atol=1e-12,rtol=0)
 expected_path=ROOT/'results/reference/metrics.csv'
 if expected_path.exists():
  expected=pd.read_csv(expected_path,dtype={"seed":str});check=m.copy();check['seed']=check.seed.astype(str);keys=['experiment_id','split','seed','method','scope','label','stratum']
  joined=expected.merge(check,on=keys,suffixes=('_expected','_actual'),validate='one_to_one');assert len(joined)==len(expected)
  for metric in ['mae_c','rmse_c']:np.testing.assert_allclose(joined[metric+'_expected'],joined[metric+'_actual'],rtol=0,atol=1e-10,equal_nan=True)
 return m

def figures(pred,m,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
 folder=out/'figures';folder.mkdir(exist_ok=True)
 def finish(fig,name,data):
  fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.pdf'));plt.close(fig);data.to_csv(folder/(name+'.csv'),index=False)
 def select(exp):return m.loc[m.experiment_id.eq(exp)&m.seed.eq('SEED_MEAN')&m.stratum.eq('all')]
 def headline(exp):return select(exp).loc[lambda x:x.scope.eq('aggregate')&x.label.eq('equal_recording')&(~x.split.str.contains('development'))]
 ridge=headline('ridge_original');methods=['persistence','temperature_ridge','telemetry_ridge','sequential_base','sequential_thermal'];t=ridge.set_index('method').loc[methods].reset_index()
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,metric in zip(axes,['mae_c','rmse_c']):ax.barh(t.method,t[metric]);ax.set_xlabel(metric.replace('_c','').upper()+' (°C)')
 fig.suptitle('Ridge sequential: paired baseline λ=10; correction λ=1, α=1\nOriginal evaluation, equal-recording weights');finish(fig,'01_ridge',t)
 neural=headline('neural_refined').set_index('method').loc[['H2','X_physics','bounded']].reset_index();fig,axes=plt.subplots(1,3,figsize=(10,3.7))
 for ax,metric in zip(axes,['mae_c','rmse_c','fraction_above_1c']):ax.bar(neural.method,neural[metric],color=['#526b85','#cf883a','#46958e']);ax.set_ylabel(metric+' (°C)' if metric!='fraction_above_1c' else 'Fraction |error| > 1°C');ax.tick_params(axis='x',rotation=15)
 fig.suptitle('Refined H2: RMSE benefit with an MAE penalty; seed means');finish(fig,'02_neural_tradeoff',neural)
 fig,axes=plt.subplots(2,2,figsize=(13,8));pieces=[]
 for ax,exp,methods in zip(axes.ravel(),['ridge_original','neural_refined','transfer','grouped'],[['sequential_thermal'],['X_physics','bounded'],['unbounded','bounded'],['unbounded','bounded']]):
  t=select(exp);t=t.loc[t.scope.eq('recording')&~t.split.str.contains('development')&t.method.isin(methods)];pieces.append(t);labels=sorted(t.label.unique())
  for i,method in enumerate(methods):
   a=t.loc[t.method.eq(method)].set_index('label').loc[labels];ax.plot(a.mae_c_minus_comparator,np.arange(len(labels))+(i-.5)*.16,'os'[i],label=method)
  ax.set_yticks(range(len(labels)));ax.set_yticklabels([x.replace('resnet_training','training') for x in labels]);ax.axvline(0,c='gray',lw=.8);ax.set_xlabel('MAE change vs named comparator (°C); negative improves');ax.set_title(exp+' vs '+('sequential_base' if exp=='ridge_original' else 'H2'));ax.legend(fontsize=7)
 finish(fig,'03_recording_changes',pd.concat(pieces))
 fig,axes=plt.subplots(1,2,figsize=(10,4));qs=['median_absolute_error_c','p90_absolute_error_c','p95_absolute_error_c','p99_absolute_error_c'];dist=[]
 for ax,exp,methods in zip(axes,['ridge_original','neural_refined'],[['sequential_base','sequential_thermal','telemetry_ridge'],['H2','X_physics','bounded']]):
  t=headline(exp).set_index('method').loc[methods];dist.append(t.reset_index())
  for method,r in t.iterrows():ax.plot([.5,.9,.95,.99],r[qs].to_numpy(float),'-o',label=method)
  ax.set_xlabel('Weighted empirical quantile');ax.set_ylabel('Absolute error (°C)');ax.set_title(exp+'; separately scored seed means');ax.legend(fontsize=8)
 finish(fig,'04_error_quantiles',pd.concat(dist))
 traces=pred.loc[pred.split.eq('heldout')&((pred.experiment_id.eq('neural_refined')&pred.seed.eq(0)&pred.method.isin(['H2','X_physics','bounded']))|(pred.experiment_id.eq('ridge_original')&pred.method.isin(['sequential_base','sequential_thermal'])))].copy()
 recordings=sorted(traces.recording_id.unique());fig,axes=plt.subplots(len(recordings),2,figsize=(13,12))
 for i,rec in enumerate(recordings):
  for j,exp in enumerate(['ridge_original','neural_refined']):
   a=traces.loc[traces.recording_id.eq(rec)&traces.experiment_id.eq(exp)];ax=axes[i,j]
   one=a.loc[a.method.eq(a.method.iloc[0])].sort_values('origin_timestamp');start=pd.to_datetime(one.origin_timestamp.iloc[0]);ax.plot((pd.to_datetime(one.origin_timestamp)-start).dt.total_seconds(),one.observed_c,c='black',label='observed future')
   for method,b in a.groupby('method'):
    b=b.sort_values('origin_timestamp');ax.plot((pd.to_datetime(b.origin_timestamp)-start).dt.total_seconds(),b.predicted_temperature_c,label=method,alpha=.8)
   ax.set_title(rec+' / '+exp);ax.set_ylabel('Temperature at t+60s (°C)');ax.set_xlabel('Origin elapsed seconds');ax.legend(fontsize=6)
 finish(fig,'05_all_recording_traces',traces)
 timing=pd.read_csv(ROOT/'provenance/prediction_timings.csv');t=headline('neural_refined');t=t.loc[t.method.isin(['H1','H2','H4','H8','H16','H32'])].copy();t['parameters']=t.method.str[1:].astype(int)*13+1
 costs=timing.loc[timing.experiment_id.eq('neural_refined')&timing.batch_size.eq(1)].groupby('method')[['median_us_per_prediction','p25_us','p75_us']].mean();t=t.merge(costs,on='method');fig,axes=plt.subplots(1,2,figsize=(10,4))
 axes[0].plot(t.parameters,t.mae_c,'o');axes[0].set_xlabel('Network trainable parameters');axes[0].set_ylabel('MAE (°C)')
 axes[1].errorbar(t.median_us_per_prediction,t.mae_c,xerr=np.array([t.median_us_per_prediction-t.p25_us,t.p75_us-t.median_us_per_prediction]),fmt='o');axes[1].set_xlabel('Recorded single-row prediction cost (µs)');axes[1].set_ylabel('MAE (°C)')
 for _,r in t.iterrows():
  axes[0].annotate(r.method,(r.parameters,r.mae_c));axes[1].annotate(r.method,(r.median_us_per_prediction,r.mae_c))
 fig.suptitle('Refined noncollapsed networks; arm64 macOS, one thread\nSaved execution only; bars: mean per-seed timing IQR, batch size 1');finish(fig,'06_capacity_cost',t)
 conv=pd.read_csv(ROOT/'provenance/convergence.csv');budget=[]
 for exp in ['budget_400','budget_2000']:
  c=conv.loc[conv.experiment_id.eq(exp)];r=headline(exp).loc[lambda x:x.method.eq('H2')].iloc[0];budget.append(dict(experiment_id=exp,network_fit_seconds=c.fit_seconds.sum(),iterations=c.iterations.sum(),evaluations=c.evaluations.sum(),unresolved=int((~c.success).sum()),mae_c=r.mae_c,rmse_c=r.rmse_c))
 t=pd.DataFrame(budget);fig,axes=plt.subplots(1,3,figsize=(11,3.8))
 for ax,metric in zip(axes,['network_fit_seconds','unresolved','mae_c']):ax.bar(t.experiment_id,t[metric]);ax.set_ylabel(metric+(' (s)' if metric=='network_fit_seconds' else ''));ax.tick_params(axis='x',rotation=15)
 fig.suptitle('Earlier collapsed-network budget study; separate saved executions\nMore computation and fewer failed fits; unchanged selected H2 errors');finish(fig,'07_training_budget',t)
 for number,exp in [('08','transfer'),('09','grouped')]:
  t=select(exp);t=t.loc[t.scope.eq('recording')];labels=sorted(t.label.unique());fig,axes=plt.subplots(1,2,figsize=(12,max(4,len(labels)*.32)))
  for ax,metric in zip(axes,['mae_c','rmse_c']):
   for i,method in enumerate(['H2','unbounded','bounded']):
    a=t.loc[t.method.eq(method)].set_index('label').loc[labels];ax.barh(np.arange(len(labels))+(i-1)*.23,a[metric],height=.23,label=method)
   ax.set_yticks(range(len(labels)));ax.set_yticklabels([x.replace('resnet_training','training') for x in labels]);ax.set_xlabel(metric+' (°C)')
  axes[0].legend();fig.suptitle(exp+': all recordings; seed means; separate evaluation scope');finish(fig,number+'_'+exp,t)
 print('Generated nine PDF/PNG figures with CSV data',flush=True)

def verify_manifest():
 path=ROOT/'provenance/package_manifest.json'
 if path.exists():
  manifest=read(path)
  for name,h in manifest.items():assert sha(ROOT/name)==h, 'Frozen package changed: '+name

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','verify','predict','plots'],default='all');args=parser.parse_args();verify_manifest()
 out=ROOT/'results/regenerated';out.mkdir(parents=True,exist_ok=True)
 if args.stage in ['all','verify','predict']:
  pred=reproduce(out);m=metrics(pred,out)
 else:pred=pd.read_parquet(out/'predictions.parquet');m=pd.read_csv(out/'metrics.csv')
 if args.stage in ['all','plots']:figures(pred,m,out)

if __name__=='__main__':main()
