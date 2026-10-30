"""Intel replication: audited acquisition, causal features, fixed-method replication."""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.optimize import minimize
SENSORS=['CPU Package [°C]','CPU Package [°C].1','CPU Package Power [W]','Total CPU Usage [%]','GPU Power [W]','GPU Core Load [%]']
FEATURES=['temperature_current_c','temperature_mean_c','temperature_slope_c_s','temperature_std_c','cpu_package_power_mean_w','gpu_power_mean_w','cpu_utilization_mean_pct','gpu_utilization_mean_pct']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def load_raw(item):
 raw=pd.read_csv(ROOT/item['path'],encoding='latin1',low_memory=False)
 keep=raw.Date.astype(str).str.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$')
 frame=raw.loc[keep].copy();metadata=raw.loc[~keep].fillna('').to_dict('records')
 full=frame.Time.astype(str).str.count(':').eq(2).all()
 ts=pd.to_datetime(frame.Date+' '+frame.Time,format='%d.%m.%Y %H:%M:%S.%f',errors='coerce') if full else pd.Series(pd.NaT,index=frame.index)
 return frame,ts,metadata

def audit():
 manifest=read(ROOT/'sources/download_manifest.json');rows=[];allstats={};overlaps=[]
 for item in manifest:
  path=ROOT/item['path'];b=path.read_bytes();assert len(b)==item['bytes'];assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['git_blob_sha1']
  f,ts,footer=load_raw(item);valid=ts.notna().all();dt=ts.diff().dt.total_seconds();stats={}
  for c in f.columns[2:]:
   v=pd.to_numeric(f[c],errors='coerce');finite=np.isfinite(v);n=int(finite.sum())
   stats[c]=dict(numeric_count=n,missing_or_nonnumeric=int((~finite).sum()),min=float(v[finite].min()) if n else None,max=float(v[finite].max()) if n else None,distinct=int(v.nunique()),adjacent_equal_fraction=float(v.eq(v.shift()).iloc[1:].mean()),median_seconds_between_value_changes=float(ts.loc[v.ne(v.shift())].diff().dt.total_seconds().median()) if valid and v.nunique()>1 else None)
  temp=pd.to_numeric(f[SENSORS[0]],errors='coerce');second=pd.to_numeric(f[SENSORS[1]],errors='coerce')
  eligible=bool(valid and not ts.duplicated().any() and (dt.dropna()>0).all() and all(c in f for c in SENSORS) and temp.notna().all())
  dates=sorted(f.Date.unique().tolist());row=dict(recording_id=item['recording_id'],workload=item['aliases'][0].split('/')[1],dates=dates,source_aliases=item['aliases'],sha256=sha(path),samples=len(f),columns=len(f.columns),duration_s=float((ts.iloc[-1]-ts.iloc[0]).total_seconds()) if valid else None,start=str(ts.iloc[0]) if valid else None,end=str(ts.iloc[-1]) if valid else None,median_interval_s=float(dt.median()) if valid else None,max_interval_s=float(dt.max()) if valid else None,duplicate_timestamps=int(ts.duplicated().sum()) if valid else None,clock_resets=int((dt<0).sum()) if valid else None,gaps_over_0_5s=int((dt>.5).sum()) if valid else None,temperature_min_c=float(temp.min()),temperature_max_c=float(temp.max()),temperature_std_c=float(temp.std()),DTS_minus_enhanced_mae_c=float((temp-second).abs().mean()),temperature_repeat_fraction=stats[SENSORS[0]]['adjacent_equal_fraction'],eligible=eligible,reason='Full timestamps and required sensors valid' if eligible else 'Ambiguous/rounded time field or duplicate/nonmonotonic timestamps; no inferred hours or repaired timestamps',device_metadata=footer)
  rows.append(row);allstats[item['recording_id']]=stats
 for i,a in enumerate(rows):
  for b in rows[i+1:]:
   if a['start'] and b['start'] and max(a['start'],b['start'])<=min(a['end'],b['end']):overlaps.append([a['recording_id'],b['recording_id']])
 result=dict(recordings=rows,sensor_statistics=allstats,overlapping_full_timestamp_captures=overlaps,license={'github':'CC BY-NC-ND 4.0 (pinned LICENSE)','figshare':'CC BY 4.0 (article metadata)','note':'Conflicting author-source notices preserved; no redistribution or license equivalence assumed.'},primary_target='CPU Package [°C] under Intel Core i7-12700: DTS; Enhanced duplicate is separately audited, never averaged',timestamp_policy='Naive source local dates; no timezone invented; ambiguous minute:second-only captures excluded',source_commit=read(ROOT/'sources/github_tree.json')['sha'])
 save(ROOT/'audit.json',result)
 pd.DataFrame([{k:v for k,v in r.items() if k not in ['device_metadata','source_aliases']} for r in rows]).to_csv(ROOT/'inventory.csv',index=False)
 print(pd.DataFrame(rows)[['recording_id','workload','dates','duration_s','eligible']].to_string(index=False),flush=True)
 return result



def prepare():
 if (ROOT/'results/model.json').exists():raise RuntimeError('Completed fit exists; use --stage finish to preserve the frozen protocol.')
 a=read(ROOT/'audit.json');manifest={r['recording_id']:r for r in read(ROOT/'sources/download_manifest.json')};rows=[];counts=[]
 for rec in a['recordings']:
  if not rec['eligible']:continue
  f,ts,_=load_raw(manifest[rec['recording_id']]);seconds=(ts-ts.iloc[0]).dt.total_seconds().to_numpy();raw=f[SENSORS].apply(pd.to_numeric,errors='coerce').to_numpy();gap=np.r_[0,np.cumsum(np.diff(seconds)>.5)]
  rejected={};n=0;previous=-1
  for tick in np.arange(30,np.floor(seconds[-1]-60)+1,1):
   i=int(np.searchsorted(seconds,tick,side='right')-1)
   if i==previous:continue
   previous=i;t=seconds[i];start=int(np.searchsorted(seconds,t-30));target=int(np.searchsorted(seconds,t+60));reason=None
   if t<30 or seconds[start]>t-30+.2:reason='history_coverage'
   elif target>=len(seconds) or seconds[target]-(t+60)>.2:reason='target_timestamp_tolerance'
   elif gap[target]!=gap[start]:reason='gap_over_0_5s'
   elif not np.isfinite(raw[start:i+1][:,[0,2,3,4,5]]).all() or not np.isfinite(raw[target,0]):reason='missing'
   if reason:rejected[reason]=rejected.get(reason,0)+1;continue
   history=raw[start:i+1];offset=seconds[start:i+1]-t;center=offset-offset.mean();temp=history[:,0]
   vals=[temp[-1],temp.mean(),float(center@(temp-temp.mean())/(center@center)),temp.std(),history[:,2].mean(),history[:,4].mean(),history[:,3].mean(),history[:,5].mean()]
   date=ts.iloc[i].strftime('%Y-%m-%d');row=dict(recording_id=rec['recording_id'],workload=rec['workload'],group=date,split='test' if date=='2025-10-01' else 'development',origin_timestamp=str(ts.iloc[i]),target_timestamp=str(ts.iloc[target]),origin_row=i,history_start_row=start,target_row=target,observed_c=raw[target,0],delta_target_c=raw[target,0]-temp[-1],achieved_horizon_s=float(seconds[target]-t),history_span_s=float(t-seconds[start]),**dict(zip(FEATURES,vals)))
   rows.append(row);n+=1
  counts.append(dict(recording_id=rec['recording_id'],origins=n,rejected=rejected))
 data=pd.DataFrame(rows);assert data.recording_id.nunique()==36;assert set(data.group)=={'2025-09-29','2025-09-30','2025-10-01'}
 data.to_parquet(ROOT/'features.parquet',index=False)
 protocol=dict(status='frozen_before_fitting_or_test_performance',history_s=30,horizon_s=60,primary_target=SENSORS[0],feature_names=FEATURES,origin_stride_s=1,origin='latest actual row at/before each one-second elapsed grid point; all histories end at actual row',history_rule='Observed rows in [origin-30,origin], oldest within 0.2s of boundary; no future filling',target_rule='First actual sample on/after origin+60, at most 0.2s late; no interpolation',max_complete_window_gap_s=.5,groups='Entire naive acquisition date, same physical machine; no independent-session assertion',development=['2025-09-29','2025-09-30'],test=['2025-10-01'],selection='Only telemetry ridge penalty {.01,.1,1,10}, leave-one-development-date-out, equal-recording MAE then balanced RMSE then stronger penalty, tolerance1e-10. No correction/H2/bound selection.',fixed_ridge=dict(lambda_value=10,correction_lambda=1,alpha=1),fixed_neural=dict(width=2,lambda_value=.03,correction_lambda=.1,alpha=.25,seeds=[0,1,2,3,4]),bound_k=1,bound='training-input equal-recording q95 of absolute complete correction, zero-only 1e-12 floor',generic='x²,x³,p*x; identical OOF targets, penalty, alpha and bounding to physics comparison',optimizer=dict(maxiter=2000,total_maxiter=8000,gtol=1e-6,ftol=1e-10,maxls=30),fit_budget=dict(intel_network=15,verification_jetson_network=1,ridge_at_most=11,correction=12),metrics='equal-recording MAE and sqrt(mean recording MSE); per-seed scores then means, no ensemble; inverse weighted ECDF quantiles',uncertainty='Only one held-out date: no group bootstrap/confidence interval; recordings and seeds not independent date replicates',transient='causal absolute 30s slope >=0.05 C/s, predeclared',adaptations='8 rather than 11 inputs; CPU package W rather than Jetson CPU/CV; timestamp tolerances for irregular Intel logging; fixed selected-settings enrichment rather than nested reselection because only two development groups',counts=counts,feature_sha256=sha(ROOT/'features.parquet'),audit_sha256=sha(ROOT/'audit.json'),implementation_sha256=sha(Path(__file__)))
 save(ROOT/'protocol.json',dict(protocol=protocol,sha256=digest(protocol)))
 data[['recording_id','workload','group','split']].drop_duplicates().to_csv(ROOT/'split_manifest.csv',index=False)
 print('PROTOCOL FROZEN',digest(protocol),data.groupby(['split','group']).size().to_dict(),flush=True)
 return data

def weights(f):return 1/(f.recording_id.nunique()*f.groupby('recording_id').recording_id.transform('size').to_numpy())
def scaler(f,cols):
 x=f[cols].to_numpy(float);w=weights(f);mu=w@x;sd=np.sqrt(w@((x-mu)**2));return mu,np.where(sd<1e-12,1,sd)
def fit_ridge(f,cols,lam):
 start=time.perf_counter();mu,sd=scaler(f,cols);z=(f[cols].to_numpy()-mu)/sd;w=weights(f);y=f.delta_target_c.to_numpy();d=np.column_stack([np.ones(len(f)),z]);pen=np.diag(np.sqrt(np.r_[0,np.full(len(cols),lam)]));b=np.linalg.lstsq(np.vstack([np.sqrt(w)[:,None]*d,pen]),np.r_[np.sqrt(w)*y,np.zeros(len(cols)+1)],rcond=None)[0]
 return dict(kind='ridge',features=cols,mean=mu.tolist(),scale=sd.tolist(),b=float(b[0]),beta=b[1:].tolist(),lambda_value=lam,training_groups=sorted(f.group.unique()),training_recordings=sorted(f.recording_id.unique()),fit_seconds=time.perf_counter()-start,parameters=len(b))
def unpack(theta,d,h=2):return theta[:d*h].reshape(d,h),theta[d*h:(d+1)*h],theta[(d+1)*h:(d+2)*h],theta[-1]
def objective(theta,z,y,w,lam):
 W,a,v,b=unpack(theta,z.shape[1]);t=np.tanh(z@W+a);e=t@v+b-y;weighted=2*w*e;back=weighted[:,None]*v*(1-t*t)
 return float(w@(e*e)+lam*((W*W).sum()+v@v)),np.r_[(z.T@back+2*lam*W).ravel(),back.sum(axis=0),t.T@weighted+2*lam*v,weighted.sum()]
def fit_network(f,seed,cols=FEATURES):
 start=time.perf_counter();mu,sd=scaler(f,cols);z=(f[cols].to_numpy()-mu)/sd;w=weights(f);y=f.delta_target_c.to_numpy();d=len(cols);rng=np.random.default_rng(seed);theta=np.r_[rng.normal(0,np.sqrt(2/(d+2)),(d,2)).ravel(),np.zeros(2),rng.normal(0,1/np.sqrt(2),2),w@y];phases=[];iterations=0
 for stage in ['initial','continuation']:
  if stage=='continuation' and result.success:break
  allowance=2000 if stage=='initial' else 8000-iterations;tic=time.perf_counter();result=minimize(objective,theta,args=(z,y,w,.03),method='L-BFGS-B',jac=True,options=dict(maxiter=allowance,gtol=1e-6,ftol=1e-10,maxls=30));theta=result.x;iterations+=int(result.nit);val,g=objective(theta,z,y,w,.03)
  phases.append(dict(stage=stage,iterations=int(result.nit),evaluations=int(result.nfev),gradient_evaluations=int(result.njev),objective=val,gradient_inf=float(abs(g).max()),success=bool(result.success),message=str(result.message),seconds=time.perf_counter()-tic))
 W,a,v,b=unpack(theta,d);return dict(kind='network',features=cols,mean=mu.tolist(),scale=sd.tolist(),W=W.tolist(),a=a.tolist(),v=v.tolist(),b=float(b),seed=seed,lambda_value=.03,training_groups=sorted(f.group.unique()),training_recordings=sorted(f.recording_id.unique()),success=bool(result.success),phases=phases,fit_seconds=time.perf_counter()-start,parameters=(d+2)*2+1)
def baseline(model,f):
 z=(f[model['features']].to_numpy()-model['mean'])/model['scale']
 return model['b']+ (z@model['beta'] if model['kind']=='ridge' else np.tanh(z@model['W']+model['a'])@model['v'])
def raw_terms(m,f,family):
 theta=f[FEATURES[0]].to_numpy()+273.15-m['T'];x=theta/m['sT'];p=(f[FEATURES[4]].to_numpy()-m['P'])/m['sP'];T=m['T']
 return np.column_stack([x*x,x*x*x,p*x]) if family=='generic' else np.column_stack([x*abs(x)**.25,(6*T*T*theta**2+4*T*theta**3+theta**4)/(4*T**3*m['sT']),p*x])
def design(m,f):
 z=(raw_terms(m,f,m['family'])-m['mean'])/m['scale'];z[:,~np.asarray(m['kept'])]=0;return z

def fit_correction(f,residual,lam,family):
 tic=time.perf_counter();w=weights(f);T=f[FEATURES[0]].to_numpy()+273.15;P=f[FEATURES[4]].to_numpy();mt=float(w@T);mp=float(w@P);st=float(np.sqrt(w@((T-mt)**2)));sp=float(np.sqrt(w@((P-mp)**2)));m=dict(T=mt,sT=st if st>1e-12 else 1,P=mp,sP=sp if sp>1e-12 else 1,family=family)
 raw=raw_terms(m,f,family);mu=w@raw;sd=np.sqrt(w@((raw-mu)**2));kept=sd>np.maximum(1e-12,1e-10*np.sqrt(w@(raw*raw)));m.update(mean=mu.tolist(),scale=np.where(kept,sd,1).tolist(),kept=kept.tolist());d=np.column_stack([np.ones(len(f)),design(m,f)]);coef=np.linalg.lstsq(np.vstack([np.sqrt(w)[:,None]*d,np.diag(np.sqrt([0,lam,lam,lam]))]),np.r_[np.sqrt(w)*residual,np.zeros(4)],rcond=1e-12)[0];m.update(c=float(coef[0]),gamma=coef[1:].tolist(),lambda_value=lam,training_groups=sorted(f.group.unique()),parameters=4,fit_seconds=time.perf_counter()-tic);return m

def quantile(x,w,q):
 i=np.argsort(x,kind='stable');return float(x[i[min(np.searchsorted(np.cumsum(w[i]/w.sum()),q),len(i)-1)]])
def metric(e,w):
 w=w/w.sum();a=abs(e);m=dict(mae_c=float(w@a),rmse_c=float(np.sqrt(w@(e*e))),bias_c=float(w@e),above_1c=float(w@(a>1)))
 for name,q in [('median',.5),('p90',.9),('p95',.95),('p99',.99)]:m[name+'_c']=quantile(a,w,q)
 return m

def validate_method():
 # Numerical derivative check and an actual original H2 fitting reproduction (11 inputs).
 rng=np.random.default_rng(17);z=rng.normal(size=(13,8));y=rng.normal(size=13);w=np.full(13,1/13);theta=rng.normal(size=21);value,g=objective(theta,z,y,w,.03);numeric=[]
 for i in range(len(theta)):
  step=np.zeros(len(theta));step[i]=1e-6;numeric.append((objective(theta+step,z,y,w,.03)[0]-objective(theta-step,z,y,w,.03)[0])/2e-6)
 np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-7)
 parent=ROOT.parent;config=read(parent/'config.json');cols=config['temperature_features']+config['telemetry_extra_features'];f=pd.read_csv(parent/'data/features.csv');f=f.loc[f.horizon_s.eq(60)&f.split.eq('development')].copy();m=fit_network(f,0,cols);saved=read(parent/'neural_capacity_refinement/results/model.json')['models']['2'][0]
 expected=np.tanh(((f[cols].to_numpy()-saved['mean'])/saved['scale'])@saved['W']+saved['a'])@saved['v']+saved['b'];diff=float(np.max(abs(expected-baseline(m,f))));assert diff<1e-8,diff
 # Original λ10 ridge replay via actual refit.
 ridge=fit_ridge(f,cols,10);old=read(parent/'results/model.json')['controlled_comparison']['frozen']['baseline'];expected=old['intercept_delta_c']+((f[cols].to_numpy()-old['feature_mean'])/old['feature_scale'])@old['coefficients_standardized'];rd=float(np.max(abs(expected-baseline(ridge,f))));assert rd<1e-10
 return dict(gradient_check_max_error=float(np.max(abs(g-numeric))),jetson_H2_refit_max_delta_difference_c=diff,jetson_ridge_refit_max_delta_difference_c=rd,source_hashes={name:sha(parent/name) for name in ['run.py','neural_capacity_refinement/run.py','bounded_physics_correction/run.py']})

def train():
 if (ROOT/'results/model.json').exists():raise RuntimeError('Saved fit exists; use --stage finish, which does not refit.')
 protocol=read(ROOT/'protocol.json');assert digest(protocol['protocol'])==protocol['sha256'];assert sha(ROOT/'features.parquet')==protocol['protocol']['feature_sha256'];assert sha(Path(__file__))==protocol['protocol']['implementation_sha256']
 f=pd.read_parquet(ROOT/'features.parquet');dev=f.loc[f.split.eq('development')];test=f.loc[f.split.eq('test')];groups=sorted(dev.group.unique());assert len(groups)==2;out=ROOT/'results';out.mkdir(exist_ok=True)
 verification=validate_method();cache={};selection=[];dev_predictions=[];models={};corrections={};bounds={};residual_records={}
 def get(kind,training,seed=-1,lam=10):
  key=kind+'|'+'|'.join(sorted(training.group.unique()))+f'|seed={seed}|lambda={lam}'
  if key not in cache:cache[key]=fit_network(training,seed) if kind=='network' else fit_ridge(training,FEATURES,lam)
  return cache[key]
 for lam in [.01,.1,1.,10.]:
  pieces=[]
  for group in groups:
   tr=dev.loc[dev.group.ne(group)];va=dev.loc[dev.group.eq(group)];m=get('ridge',tr,lam=lam);e=baseline(m,va)-va.delta_target_c.to_numpy();pieces.extend([dict(recording_id=r,mae_c=float(abs(e[va.recording_id.eq(r)]).mean()),mse=float((e[va.recording_id.eq(r)]**2).mean())) for r in va.recording_id.unique()])
  selection.append(dict(lambda_value=lam,mae_c=float(np.mean([x['mae_c'] for x in pieces])),rmse_c=float(np.sqrt(np.mean([x['mse'] for x in pieces])))))
 candidate=pd.DataFrame(selection);candidate=candidate.loc[candidate.mae_c<=candidate.mae_c.min()+1e-10];candidate=candidate.loc[candidate.rmse_c<=candidate.rmse_c.min()+1e-10];chosen=float(candidate.sort_values('lambda_value',ascending=False).iloc[0].lambda_value)
 models['temperature_ridge']=fit_ridge(dev,FEATURES[:4],1);models['tuned_telemetry']=get('ridge',dev,lam=chosen);models['ridge_fixed']=get('ridge',dev,lam=10.)
 for kind,seeds,lam,alpha in [('ridge',[-1],1.,1.),('network',list(range(5)),.1,.25)]:
  for seed in seeds:
   tic=time.perf_counter();res=np.full(len(dev),np.nan);sources=[]
   for group in groups:
    tr=dev.loc[dev.group.ne(group)];va=dev.loc[dev.group.eq(group)];m=get(kind,tr,seed,10. if kind=='ridge' else .03);assert group not in m['training_groups'];mask=dev.group.eq(group).to_numpy();values=baseline(m,va);res[mask]=va.delta_target_c.to_numpy()-values;sources.append(dict(excluded_group=group,training_groups=m['training_groups'],training_recordings=m['training_recordings']))
    for rec in va.recording_id.unique():
     ix=va.recording_id.eq(rec).to_numpy();dev_predictions.append(dict(method='ridge_fixed' if kind=='ridge' else 'H2',seed=seed,recording_id=rec,group=group,**metric(values[ix]-va.delta_target_c.to_numpy()[ix],np.ones(ix.sum()))))
   receiver=get(kind,dev,seed,10. if kind=='ridge' else .03);key='ridge_fixed' if kind=='ridge' else f'H2_seed{seed}';models[key]=receiver;frozen=digest(receiver);residual_records[key]=dict(sources=sources,residual_sha256=hashlib.sha256(res.tobytes()).hexdigest(),inclusive_wall_seconds=time.perf_counter()-tic)
   for family in ['physics','generic']:
    c=fit_correction(dev,res,lam,family);corrections[key+'_'+family]=c;delta=alpha*(c['c']+design(c,dev)@c['gamma']);q=quantile(abs(delta),weights(dev),.95);bounds[key+'_'+family]=q if q!=0 else 1e-12
   assert digest(receiver)==frozen
   print('TRAINED',key,'success',receiver.get('success',True),flush=True)
 assert len([v for v in cache.values() if v['kind']=='network'])==15
 frozen=dict(protocol_sha256=protocol['sha256'],models=models,corrections=corrections,bounds=bounds,selected_telemetry_lambda=chosen,selection=selection,residual_provenance=residual_records,all_fits=cache,verification=verification,development_validation=dev_predictions)
 artifact=dict(stage='frozen_before_test_predictions',frozen=frozen,freeze_sha256=digest(frozen));save(out/'model.json',artifact);print('MODELS FROZEN; evaluating test once',artifact['freeze_sha256'],flush=True)
 rows=[]
 def record(method,seed,pred,base,orig,applied,comparator):
  p=test[['recording_id','workload','group','origin_timestamp','target_timestamp','observed_c','delta_target_c',FEATURES[0],FEATURES[2]]].copy();p['method']=method;p['seed']=seed;p['comparator']=comparator;p['predicted_delta_c']=pred;p['predicted_c']=test[FEATURES[0]].to_numpy()+pred;p['baseline_delta_c']=base;p['original_correction_c']=orig;p['applied_correction_c']=applied;p['inside_training_range']=test[FEATURES[0]].between(dev[FEATURES[0]].min(),dev[FEATURES[0]].max());p['transient']=test[FEATURES[2]].abs().ge(.05);rows.append(p)
 zero=np.zeros(len(test));record('persistence',-1,zero,zero,zero,zero,'ridge_fixed')
 for name in ['temperature_ridge','tuned_telemetry','ridge_fixed']:
  b=baseline(models[name],test);record(name,-1,b,b,zero,zero,'ridge_fixed')
 for key in ['ridge_fixed']+[f'H2_seed{s}' for s in range(5)]:
  seed=-1 if key=='ridge_fixed' else int(key[-1]);alpha=1 if seed==-1 else .25;base=baseline(models[key],test);name='ridge' if seed==-1 else 'H2'
  if seed!=-1:record('H2',seed,base,base,zero,zero,'H2')
  for family in ['physics','generic']:
   c=corrections[key+'_'+family];original=alpha*(c['c']+design(c,test)@c['gamma']);record(name+'_'+family,seed,base+original,base,original,original,'ridge_fixed' if seed==-1 else 'H2')
   if seed!=-1:
    s=bounds[key+'_'+family];applied=s*np.tanh(original/s);record(name+'_'+family+'_bounded',seed,base+applied,base,original,applied,'H2')
 pred=pd.concat(rows,ignore_index=True);pred.to_csv(out/'predictions.csv',index=False)
 result=summarize(pred);result.to_csv(out/'metrics.csv',index=False)
 artifact['stage']='evaluated_without_settings_revision';artifact['coverage']=dict(training_temperature_range_c=[float(dev[FEATURES[0]].min()),float(dev[FEATURES[0]].max())],test_temperature_range_c=[float(test[FEATURES[0]].min()),float(test[FEATURES[0]].max())],test_outside_fraction=float(weights(test)@(~test[FEATURES[0]].between(dev[FEATURES[0]].min(),dev[FEATURES[0]].max())).to_numpy()));artifact['run_environment']=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,numerical_threads=1)
 # Small, declared prediction-cost measurement: 5 repeated complete heldout batches; no timing-based selection.
 costs=[]
 for key,m in models.items():
  samples=[]
  for rep in range(5):
   start=time.perf_counter();baseline(m,test);samples.append(time.perf_counter()-start)
  costs.append(dict(method=key,batch_size=len(test),repetitions=5,seconds=samples,median_seconds=float(np.median(samples)),scope='baseline vectorized inference only; excludes correction, data IO and feature extraction'))
 artifact['prediction_costs']=costs;save(out/'model.json',artifact);report(result,pred,artifact)
 preservation=read(ROOT/'preservation.json');after={name:sha(ROOT.parent/name) for name in preservation['before']};assert after==preservation['before'];preservation.update(after=after,complete=True,changed=[]);save(ROOT/'preservation.json',preservation)
 print(result.loc[result.seed.eq('SEED_MEAN')&result.recording_id.eq('EQUAL_RECORDING')&result.stratum.eq('all'),['method','mae_c','rmse_c']].to_string(index=False),flush=True)

def summarize(pred):
 rows=[]
 for (method,seed),p in pred.groupby(['method','seed']):
  e=p.predicted_delta_c.to_numpy()-p.delta_target_c.to_numpy();be=p.baseline_delta_c.to_numpy()-p.delta_target_c.to_numpy();corr=p.applied_correction_c.to_numpy();w=weights(p)
  for rid in sorted(p.recording_id.unique())+['EQUAL_RECORDING']:
   rec=np.ones(len(p),bool) if rid=='EQUAL_RECORDING' else p.recording_id.eq(rid).to_numpy()
   for stratum,mask in [('all',rec),('inside',rec&p.inside_training_range.to_numpy()),('outside',rec&~p.inside_training_range.to_numpy()),('transient',rec&p.transient.to_numpy()),('nontransient',rec&~p.transient.to_numpy())]:
    row=dict(method=method,seed=int(seed),recording_id=rid,stratum=stratum,n=int(mask.sum()),comparator=p.comparator.iloc[0],group=p.group.iloc[0],workload=p.workload.iloc[0])
    if mask.any():
     v=w[mask]/w[mask].sum();row.update(metric(e[mask],v));row.update(correction_mean_c=float(v@corr[mask]),correction_abs_mean_c=float(v@abs(corr[mask])),wrong_direction=float(v@((be[mask]*corr[mask])>0)),harmful_overshoot=float(v@(((be[mask]*e[mask])<0)&(abs(e[mask])>abs(be[mask])))))
    rows.append(row)
 m=pd.DataFrame(rows);keys=['method','recording_id','stratum','comparator','group','workload'];num=[c for c in m.select_dtypes(include=np.number) if c not in ['seed','n']];mean=m.groupby(keys)[num].mean().reset_index();mean['seed']='SEED_MEAN';sd=m.groupby(keys)[num].std().reset_index();sd['seed']='SEED_SD';m=pd.concat([m,mean,sd],ignore_index=True);lookup=m.set_index(['method','seed','recording_id','stratum'])
 for metric_name in ['mae_c','rmse_c']:
  diff=[];pct=[]
  for _,r in m.iterrows():
   key=(r.comparator,r.seed,r.recording_id,r.stratum);value=lookup.loc[key,metric_name] if r.seed!='SEED_SD' else np.nan;diff.append(r[metric_name]-value);pct.append(100*(r[metric_name]-value)/value if value else np.nan)
  m[metric_name+'_minus_paired']=diff;m[metric_name+'_pct_change']=pct
 return m

def report(m,p,a):
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 out=ROOT/'results';head=m.loc[m.seed.eq('SEED_MEAN')&m.recording_id.eq('EQUAL_RECORDING')&m.stratum.eq('all')];recs=sorted(p.recording_id.unique());fig,axes=plt.subplots(4,2,figsize=(13,12));axes=axes.ravel()
 axes[0].barh(head.method,head.mae_c);axes[0].set_title('Held-out image-captioning: equal-recording MAE (°C)')
 for ax,rid in zip(axes[1:],recs):
  part=p.loc[p.recording_id.eq(rid)&p.seed.eq(0)];first=part.loc[part.method.eq('H2')];start=pd.to_datetime(first.origin_timestamp.iloc[0]);ax.plot((pd.to_datetime(first.origin_timestamp,format='mixed')-start).dt.total_seconds(),first.observed_c,c='black',label='observed')
  for method in ['H2','H2_physics','H2_physics_bounded','H2_generic_bounded']:
   sub=part.loc[part.method.eq(method)];ax.plot((pd.to_datetime(sub.origin_timestamp,format='mixed')-start).dt.total_seconds(),sub.predicted_c,label=method,alpha=.8)
  ax.set_title(rid);ax.set_ylabel('CPU package at t+60s (°C)');ax.set_xlabel('Origin elapsed s');ax.legend(fontsize=6)
 fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 text=['# Intel i7-12700 replication','', 'Fixed-settings replication on independent hardware data; not frozen Jetson-weight transfer. All same-date captures stay together. Development: 29–30 September 2025; untouched final test: 1 October, seven image-captioning captures. Only one test date: no valid session-level confidence interval or broad workload-generalization claim.','', '| Method | MAE °C | RMSE °C |','|---|---:|---:|']
 for _,r in head.iterrows():text.append(f'| {r.method} | {r.mae_c:.6f} | {r.rmse_c:.6f} |')
 text+=['','Four captures excluded before fitting: three hourless rounded timestamps and one backward clock jump. Forty unique files were downloaded by pinned Git blob; aliases are not extra sessions. Valid durations are approximately 15–16 minutes; median logging intervals 93–99ms do not establish sensor refresh rates. Repeated readings and sensor-channel differences are recorded in audit.json.','', 'Eight causal inputs replace eleven Jetson inputs: four package-temperature histories, CPU package/GPU board power, CPU/GPU utilization. Missing rails/core counts are omitted. DTS package temperature is the predeclared target; Enhanced is not mixed into it. History is 30 seconds, origins sampled causally at 1-second stride, exact observed target chosen 60–60.2 seconds ahead, no interpolation and no >0.5s gap across history-to-target.','', 'Ridge fixed λ=10, correction λ=1/α=1; H2 λ=.03, correction λ=.1/α=.25, five seeds. OOF residuals exclude whole development dates. Generic (x²,x³,px) corrections use identical residuals/settings. Bounds use only training-input q95. Only telemetry ridge λ is tuned over four choices, using development dates. Original nested correction reselection is not feasible with two development groups; this distinction is explicit. Coefficients are learned residual adjustments, not measured physical constants.','', 'Source documentation: authors’ GitHub at pinned commit and Figshare DOI 10.6084/m9.figshare.31654879; paper DOI 10.1038/s41597-026-07496-6. GitHub/ Figshare license notices differ and are retained. See METHODOLOGY.md, protocol.json and PROGRESS.md for continuation. Existing Jetson artifacts are hash-preserved.']
 (out/'summary.md').write_text('\n'.join(text)+'\n')

def finish():
 out=ROOT/'results';a=read(out/'model.json');frozen=a['frozen'];assert digest(frozen)==a['freeze_sha256']
 protocol=read(ROOT/'protocol.json');assert digest(protocol['protocol'])==protocol['sha256'];assert sha(ROOT/'features.parquet')==protocol['protocol']['feature_sha256']
 data=pd.read_parquet(ROOT/'features.parquet');test=data.loc[data.split.eq('test')];pred=pd.read_csv(out/'predictions.csv',float_precision='round_trip');m=pd.read_csv(out/'metrics.csv');head=m.loc[m.seed.eq('SEED_MEAN')&m.recording_id.eq('EQUAL_RECORDING')&m.stratum.eq('all')].set_index('method')
 costs=[];maxdiff=0.
 for (method,seed),part in pred.groupby(['method','seed']):
  assert part.origin_timestamp.tolist()==test.origin_timestamp.tolist()
  if method=='persistence':predict=lambda:np.zeros(len(test));parameters=0
  else:
   key=f'H2_seed{seed}' if method.startswith('H2') else 'ridge_fixed' if method.startswith('ridge_') else method
   model=frozen['models'][key];parameters=model['parameters']
   if 'physics' in method or 'generic' in method:
    family='physics' if 'physics' in method else 'generic';c=frozen['corrections'][key+'_'+family];alpha=.25 if method.startswith('H2') else 1.;bound=frozen['bounds'][key+'_'+family];parameters+=4
    def predict():
     delta=alpha*(c['c']+design(c,test)@c['gamma']);return baseline(model,test)+(bound*np.tanh(delta/bound) if method.endswith('bounded') else delta)
   else:predict=lambda:baseline(model,test)
  value=predict();maxdiff=max(maxdiff,float(np.max(abs(value-part.predicted_delta_c.to_numpy()))));assert maxdiff<1e-10
  samples=[]
  for repeat in range(5):
   start=time.perf_counter();predict();samples.append(time.perf_counter()-start)
  costs.append(dict(method=method,seed=int(seed),parameters=parameters,batch_size=len(test),repetitions=5,seconds=samples,median_seconds=float(np.median(samples)),p25_seconds=float(np.quantile(samples,.25)),p75_seconds=float(np.quantile(samples,.75)),scope='Complete baseline+correction+bound inference on prepared features; excludes feature extraction and IO; not used in selection'))
 a['complete_prediction_costs']=costs;a['saved_replay_max_error_c']=maxdiff
 wins={}
 for method in head.index:
  rows=m.loc[m.method.eq(method)&m.seed.eq('SEED_MEAN')&m.stratum.eq('all')&m.recording_id.ne('EQUAL_RECORDING')]
  wins[method]=dict(mae_wins=int((rows.mae_c_minus_paired<0).sum()),rmse_wins=int((rows.rmse_c_minus_paired<0).sum()),denominator=len(rows),comparator=head.loc[method,'comparator'])
 a['recording_win_counts']=wins
 a['interpretation']='Physics improves its fixed paired receivers modestly, but matched sequential generic corrections perform better. Tuned telemetry ridge outperforms fixed ridge physics. One held-out acquisition date/workload cannot establish general or statistically independent multi-session superiority.'
 a['jetson_comparison']=dict(scope='Separate independent hardware evaluations; values not pooled, predictor dimension/target/split differ',jetson_fixed_ridge_paired=[.350547,.432464],jetson_ridge_physics=[.298300,.371303],jetson_H2=[.278323,.428505],jetson_H2_physics=[.291615,.401240],jetson_H2_bounded=[.282923,.409968],intel={name:dict(mae_c=float(head.loc[name,'mae_c']),rmse_c=float(head.loc[name,'rmse_c'])) for name in ['ridge_fixed','ridge_physics','H2','H2_physics','H2_physics_bounded']})
 a['report_only_repair']=dict(reason='Mixed fractional-second timestamp strings in figure parser; no changes to fitting, frozen models, predictions, scores or settings.',implementation_sha256=sha(Path(__file__)))
 save(out/'model.json',a);report(m,pred,a)
 text=(out/'summary.md').read_text();text+='\n**Outcome:** ridge physics MAE improves 6.69% against its fixed λ=10 receiver, but tuned telemetry ridge is stronger. H2 physics improves MAE 0.26% and RMSE 0.43%; generic correction improves both more. Bounding reduces these small neural gains. These results support limited nonlinear correction benefits, not superiority of the physics feature forms. All 15 Intel H2 fits converged. Full recording results, transient/range strata, exact paired differences and seed variability remain in metrics.csv; wins, computational measurements and separate Jetson comparison are in model.json.\n'
 (out/'summary.md').write_text(text)
 preservation=read(ROOT/'preservation.json');after={name:sha(ROOT.parent/name) for name in preservation['before']};assert after==preservation['before'];preservation.update(after=after,complete=True,changed=[]);save(ROOT/'preservation.json',preservation)
 print('Saved models replayed, figures repaired; original predictions and scores unchanged. Protected files:',len(after),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','prepare','train','finish'],default='finish');args=parser.parse_args()
 {'audit':audit,'prepare':prepare,'train':train,'finish':finish}[args.stage]()
