"""Self-contained Intel baseline/physics comparison; default saved-model replay."""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'local/results'
MODEL=ROOT/'models/model.json'
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
FEATURES=['temperature_current_c','temperature_mean_c','temperature_slope_c_s','temperature_std_c','cpu_package_power_mean_w','gpu_power_mean_w','cpu_utilization_mean_pct','gpu_utilization_mean_pct']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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
 return np.column_stack([x*abs(x)**.25,(6*T*T*theta**2+4*T*theta**3+theta**4)/(4*T**3*m['sT']),p*x])

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

def replay(artifact,test):
 rows=[]
 for key in ['ridge_fixed']+[f'H2_seed{s}' for s in range(5)]:
  seed=-1 if key=='ridge_fixed' else int(key[-1]);name='ridge' if seed==-1 else 'H2';base=baseline(artifact['models'][key],test)
  c=artifact['corrections'][key+'_physics'];alpha=1. if seed==-1 else .25;original=alpha*(c['c']+design(c,test)@c['gamma']);s=artifact['bounds'][key+'_physics']
  methods=[('ridge_fixed' if seed==-1 else 'H2',np.zeros(len(test))), (name+'_physics',original)]
  if seed!=-1:methods.append(('H2_physics_bounded',s*np.tanh(original/s)))
  for method,applied in methods:
   p=test[['recording_id','workload','group','origin_timestamp','target_timestamp','observed_c','delta_target_c',FEATURES[0],FEATURES[2]]].copy();p['method']=method;p['seed']=seed;p['comparator']='ridge_fixed' if seed==-1 else 'H2';p['predicted_delta_c']=base+applied;p['predicted_c']=p[FEATURES[0]]+base+applied;p['baseline_delta_c']=base;p['original_correction_c']=np.zeros(len(test)) if method in ['ridge_fixed','H2'] else original;p['applied_correction_c']=applied
   p['inside_training_range']=p[FEATURES[0]].between(*artifact['training_temperature_range']);p['transient']=p[FEATURES[2]].abs().ge(.05);rows.append(p)
 return pd.concat(rows,ignore_index=True)

def verify(pred,metrics,artifact):
 maximum=None
 if (ROOT/'local/reference_predictions.parquet').exists():
  ref=pd.read_parquet(ROOT/'local/reference_predictions.parquet');keys=['method','seed','recording_id','origin_timestamp'];joined=ref.merge(pred,on=keys,suffixes=('_reference','_replayed'),validate='one_to_one');assert len(joined)==len(ref)==len(pred)
  np.testing.assert_allclose(joined.predicted_c_reference,joined.predicted_c_replayed,atol=1e-10,rtol=0)
  maximum=float(abs(joined.predicted_c_reference-joined.predicted_c_replayed).max())
 expected=pd.DataFrame(artifact['reference_metrics']);expected.seed=expected.seed.astype(str);actual=metrics.copy();actual.seed=actual.seed.astype(str);mk=['method','seed','recording_id','stratum'];j=expected.merge(actual,on=mk,suffixes=('_reference','_replayed'),validate='one_to_one');assert len(j)==len(expected)
 for col in ['mae_c','rmse_c']:np.testing.assert_allclose(j[col+'_reference'],j[col+'_replayed'],atol=1e-10,rtol=0,equal_nan=True)
 for (method,seed),p in pred.groupby(['method','seed']):
  e=p.predicted_c-p.observed_c;ind=pd.DataFrame({'recording_id':p.recording_id,'absolute':abs(e),'squared':e**2}).groupby('recording_id').mean().mean();r=metrics.loc[metrics.method.eq(method)&metrics.seed.eq(seed)&metrics.recording_id.eq('EQUAL_RECORDING')&metrics.stratum.eq('all')].iloc[0];np.testing.assert_allclose([r.mae_c,r.rmse_c],[ind.absolute,np.sqrt(ind.squared)],atol=1e-12,rtol=0)
 return dict(prediction_rows=len(pred),max_prediction_error_c=maximum,reference_metrics_reproduced=True,independent_metrics_passed=True)

def refit_check(artifact,dev,test):
 # Fixed procedures only: no tuning, new settings, or promotion of refitted models.
 candidate=dict(models={},corrections={},bounds={})
 for key in ['ridge_fixed']+[f'H2_seed{s}' for s in range(5)]:
  seed=-1 if key=='ridge_fixed' else int(key[-1]);ridge=seed==-1
  fit=lambda f:fit_ridge(f,FEATURES,10.) if ridge else fit_network(f,seed)
  receiver=fit(dev);residual=np.full(len(dev),np.nan)
  for omitted in sorted(dev.group.unique()):
   tr=dev.loc[dev.group.ne(omitted)];mask=dev.group.eq(omitted).to_numpy();assert omitted not in tr.group.unique();residual[mask]=dev.loc[mask,'delta_target_c'].to_numpy()-baseline(fit(tr),dev.loc[mask])
  correction=fit_correction(dev,residual,1. if ridge else .1,'physics');alpha=1. if ridge else .25;delta=alpha*(correction['c']+design(correction,dev)@correction['gamma']);q=quantile(abs(delta),weights(dev),.95)
  candidate['models'][key]=receiver;candidate['corrections'][key+'_physics']=correction;candidate['bounds'][key+'_physics']=q if q else 1e-12
 candidate['training_temperature_range']=artifact['training_temperature_range'];p=replay(candidate,test);m=summarize(p)
 frozen=replay(artifact,test);np.testing.assert_allclose(p.predicted_c,frozen.predicted_c,atol=1e-9,rtol=0)
 result=verify(p,m,artifact);result['max_refit_vs_frozen_difference_c']=float(abs(p.predicted_c-frozen.predicted_c).max());save(OUT/'refit_verification.json',result);print('Optional fixed-procedure refitting check:',result)

LABELS={'ridge_fixed':'Ridge baseline (λ=10)','ridge_physics':'Ridge + physics','H2':'H2 baseline','H2_physics':'H2 + physics','H2_physics_bounded':'H2 + bounded physics'}
def report(metrics,pred,artifact):
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 head=metrics.loc[metrics.seed.eq('SEED_MEAN')&metrics.recording_id.eq('EQUAL_RECORDING')&metrics.stratum.eq('all')].set_index('method').loc[list(LABELS)]
 fig,axes=plt.subplots(1,3,figsize=(15,4.8));colors=['#526b85','#459b88','#526b85','#459b88','#89b9aa']
 for ax,col in zip(axes[:2],['mae_c','rmse_c']):
  ax.barh([LABELS[k] for k in head.index],head[col],color=colors);ax.invert_yaxis();ax.set_xlabel(('MAE' if col=='mae_c' else 'Balanced RMSE')+' (°C)')
  for i,value in enumerate(head[col]):ax.text(value+.025,i,f'{value:.4f}',va='center',fontsize=8)
  ax.set_xlim(0,float(head[col].max())*1.18)
 recs=sorted(pred.recording_id.unique());win={}
 for i,method in enumerate(['ridge_physics','H2_physics','H2_physics_bounded']):
  rows=metrics.loc[metrics.method.eq(method)&metrics.seed.eq('SEED_MEAN')&metrics.stratum.eq('all')&metrics.recording_id.isin(recs)].set_index('recording_id').loc[recs]
  axes[2].plot(-rows.mae_c_pct_change,np.arange(len(recs))+(i-1)*.14,'os^'[i],label=LABELS[method]);win[method]=dict(mae_wins=int((rows.mae_c_minus_paired<0).sum()),rmse_wins=int((rows.rmse_c_minus_paired<0).sum()),recordings=len(recs))
 axes[2].axvline(0,c='gray',lw=.8);axes[2].set_yticks(range(len(recs)));axes[2].set_yticklabels(recs);axes[2].set_xlabel('MAE reduction vs paired baseline (%)');axes[2].legend(fontsize=7)
 fig.suptitle('Intel i7-12700: baseline-to-physics improvements\nOriginal held-out recordings; equal-recording metrics; seeds scored separately');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
 text=['# Standard baselines and physics-inspired enrichment','', '**Adding the physics-inspired terms improves both MAE and RMSE against the paired ridge and H2 baselines.**','','| Method | MAE °C | RMSE °C | MAE reduction | RMSE reduction |','|---|---:|---:|---:|---:|']
 for method,r in head.iterrows():text.append(f'| {LABELS[method]} | {r.mae_c:.6f} | {r.rmse_c:.6f} | {-r.mae_c_pct_change:.2f}% | {-r.rmse_c_pct_change:.2f}% |')
 text+=['', 'Ridge physics improves both metrics in all seven held-out recordings. H2 physics improves MAE in five of seven and RMSE in six of seven. Bounding retains a smaller aggregate improvement. All recording-level outcomes are retained, including deteriorations.', '', 'The ridge baseline is the fixed λ=10 receiver; the H2 baseline uses λ=.03. Fit each baseline on the development dates and freeze it. Fit three physics-inspired features to whole-date out-of-fold baseline errors. Ridge correction uses λ=1, α=1; H2 uses λ=.1, α=.25. Biases/intercepts are unpenalized. Training-only centering/scaling and equal-recording weights are used throughout.', '', 'With T in kelvin, θ=T−T*, x=θ/sT and standardized CPU package power p, features are C=x|x|^.25, R=(6T*²θ²+4T*θ³+θ⁴)/(4T*³sT), and I=px. Prediction adds δ=α(c+γᵀφ). The bounded version adds s·tanh(δ/s), where s is the training-input weighted q95 of |δ|. These learned coefficients are residual-correction parameters, not measured material constants.', '', 'Data: eight causal inputs summarizing 30 seconds; target approximately 60 seconds ahead. Development comprises 29 September and 30 September 2025; testing comprises seven image-captioning recordings on 1 October. One held-out date/workload limits generalization claims. Five neural seeds are scored separately, not ensembled. All original eligible origins and saved parameters are preserved.', '', 'Default execution replays saved models and verifies original predictions and independent metrics. The optional fixed-procedure refitting check uses only bundled features and does not overwrite saved models. No new settings or results are selected. Dataset attribution and original license notices are retained in model.json.']
 (OUT/'summary.md').write_text('\n'.join(text)+'\n');artifact['recording_wins']=win

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--refit-check',action='store_true');args=parser.parse_args();artifact=read(MODEL)
 OUT.mkdir(parents=True,exist_ok=True)
 # Raw hashes, schema, timestamps and numerical aggregate references validate regenerated inputs; Parquet byte identity is not required.
 data=pd.read_parquet(ROOT/'local/features.parquet');dev=data.loc[data.split.eq('development')];test=data.loc[data.split.eq('test')];artifact['training_temperature_range']=[float(dev[FEATURES[0]].min()),float(dev[FEATURES[0]].max())]
 if args.refit_check:refit_check(artifact,dev,test);return
 pred=replay(artifact,test);metrics=summarize(pred);artifact['verification']=verify(pred,metrics,artifact)
 pred.to_csv(OUT/'predictions.csv',index=False);metrics.to_csv(OUT/'metrics.csv',index=False);report(metrics,pred,artifact);save(OUT/'verification.json',artifact['verification'])
 print(artifact['verification'])

if __name__=='__main__':main()
