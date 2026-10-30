import numpy as np
import time
from scipy.optimize import minimize
FEATURES=['temperature_current_c', 'temperature_mean_c', 'temperature_slope_c_s', 'temperature_std_c', 'cpu_package_power_mean_w', 'gpu_power_mean_w', 'cpu_utilization_mean_pct', 'gpu_utilization_mean_pct']

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
