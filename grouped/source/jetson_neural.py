import numpy as np
import time
from scipy.optimize import minimize
FEATURES=['cpu_current_c', 'cpu_history_mean_c', 'cpu_history_slope_c_per_s', 'cpu_history_std_c', 'power_VDD_GPU_SOC_mean_mw', 'power_VDD_CPU_CV_mean_mw', 'power_VIN_SYS_5V0_mean_mw', 'power_VDDQ_VDD2_1V8AO_mean_mw', 'gpu_mean_pct', 'cpu_active_mean_pct', 'active_core_count']
CFG={'optimizer': {'maxiter': 2000, 'gtol': 1e-06, 'ftol': 1e-10, 'maxls': 30}, 'continuation': {'total_maxiter': 8000, 'policy': 'One parameter restart for every unsuccessful initial fit; all widths, seeds and selection subsets; original tolerances unchanged', 'competitive_margin_mae_c': 0.01}}

def weights(frame):
    # Provenance: parent run.py weights (equal total weight per recording window).
    return 1/(frame.window_id.nunique()*frame.groupby('window_id').window_id.transform('size').to_numpy())

def standardize_fit(frame):
    raw=frame[FEATURES].to_numpy(float); w=weights(frame)
    mean=w@raw; sd=np.sqrt(w@((raw-mean)**2))
    return dict(mean=mean.tolist(),scale=np.where(sd<1e-12,1,sd).tolist())

def unpack(theta,h):
    return theta[:11*h].reshape(11,h),theta[11*h:12*h],theta[12*h:13*h],theta[-1]

def objective(theta,z,y,w,h,lam):
    W,a,v,b=unpack(theta,h)
    t=np.tanh(z@W+a); e=t@v+b-y
    weighted=2*w*e; back=weighted[:,None]*v*(1-t*t)
    grad=np.r_[(z.T@back+2*lam*W).ravel(),back.sum(axis=0),t.T@weighted+2*lam*v,weighted.sum()]
    return float(w@(e*e)+lam*((W*W).sum()+v@v)),grad

def fit_network(frame,h,lam,seed):
    start=time.perf_counter(); scaling=standardize_fit(frame)
    raw=frame[FEATURES].to_numpy(float); z=(raw-scaling['mean'])/scaling['scale']
    w=weights(frame); y=frame.delta_target_c.to_numpy()
    rng=np.random.default_rng(seed)
    theta=np.r_[rng.normal(0,np.sqrt(2/(11+h)),(11,h)).ravel(),np.zeros(h),
                rng.normal(0,1/np.sqrt(h),h),w@y]
    phases=[]; total_iterations=0
    for stage in ('initial','continuation'):
        if stage=='continuation' and fit.success:break
        allowance=CFG['optimizer']['maxiter'] if stage=='initial' else CFG['continuation']['total_maxiter']-total_iterations
        if allowance<=0:break
        phase_start=time.perf_counter()
        fit=minimize(objective,theta,args=(z,y,w,h,lam),method='L-BFGS-B',jac=True,options=dict(CFG['optimizer'],maxiter=allowance))
        elapsed=time.perf_counter()-phase_start
        total_iterations+=int(fit.nit);theta=fit.x
        val,grad=objective(theta,z,y,w,h,lam)
        phases.append(dict(stage=stage,allowance=allowance,iterations=int(fit.nit),evaluations=int(fit.nfev),gradient_evaluations=int(fit.njev),
                           objective=val,gradient_inf=float(np.max(np.abs(grad))),success=bool(fit.success),status=int(fit.status),message=str(fit.message),wall_seconds=elapsed))
    W,a,v,b=unpack(theta,h)
    model=dict(width=h,lambda_value=lam,seed=seed,**scaling,W=W.tolist(),a=a.tolist(),v=v.tolist(),b=float(b),
               training_windows=sorted(frame.window_id.unique()),training_n=len(frame),fit_seconds=time.perf_counter()-start,
               objective=val,gradient_inf=float(np.max(np.abs(grad))),success=bool(fit.success),status=int(fit.status),message=str(fit.message),
               iterations=total_iterations,evaluations=sum(p['evaluations'] for p in phases),gradient_evaluations=sum(p['gradient_evaluations'] for p in phases),
               postfit_check_objective_gradient_evaluations=len(phases),phases=phases,continuation_seconds=sum(p['wall_seconds'] for p in phases if p['stage']=='continuation'))
    model['training_diagnostics']=diagnose(model,frame)
    return model

def prepare_network(model):
    return {k:np.asarray(model[k]) for k in ('mean','scale','W','a','v','b')}

def network_delta(model,raw):
    return np.tanh(((raw-model['mean'])/model['scale'])@model['W']+model['a'])@model['v']+model['b']

def raw_physics(model,raw):
    # Provenance: parent run.py radiation_feature/enrichment_inputs/fit_centered_physics.
    # Same Kelvin reference, CPU/CV watts, stable polynomial, and no projection.
    theta=raw[:,0]+273.15-model['temperature_reference_k']; s=model['temperature_scale_k']
    x=theta/s; p=(raw[:,5]/1000-model['power_mean_w'])/model['power_scale_w']
    T=model['temperature_reference_k']
    radiation=(6*T**2*theta**2+4*T*theta**3+theta**4)/(4*T**3*s)
    return np.column_stack((x*np.abs(x)**.25,radiation,p*x))

def fit_physics_scaling(frame):
    w=weights(frame); raw=frame[FEATURES].to_numpy(float)
    t=raw[:,0]+273.15; p=raw[:,5]/1000
    tm=float(w@t); pm=float(w@p)
    ts=float(np.sqrt(w@((t-tm)**2))); ps=float(np.sqrt(w@((p-pm)**2)))
    model=dict(temperature_reference_k=tm,temperature_scale_k=ts if ts>1e-12 else 1,
               power_mean_w=pm,power_scale_w=ps if ps>1e-12 else 1)
    phi=raw_physics(model,raw); mu=w@phi; sd=np.sqrt(w@((phi-mu)**2))
    kept=sd>np.maximum(1e-12,1e-10*np.sqrt(w@(phi*phi)))
    model.update(mean=mu.tolist(),scale=np.where(kept,sd,1).tolist(),kept=kept.tolist())
    return model

def physics_design(model,raw):
    phi=(raw_physics(model,raw)-model['mean'])/model['scale']
    phi[:,~np.asarray(model['kept'],dtype=bool)]=0
    return phi

def fit_correction(frame,residual,lam):
    start=time.perf_counter(); model=fit_physics_scaling(frame)
    phi=physics_design(model,frame[FEATURES].to_numpy(float)); w=weights(frame)
    d=np.column_stack((np.ones(len(frame)),phi))
    # Same augmented least-squares ridge objective as parent; free intercept.
    augmented=np.vstack((np.sqrt(w)[:,None]*d,np.diag(np.sqrt([0,lam,lam,lam]))))
    co=np.linalg.lstsq(augmented,np.r_[np.sqrt(w)*residual,np.zeros(4)],rcond=1e-12)[0]
    model.update(c=float(co[0]),gamma=co[1:].tolist(),lambda_value=lam,training_windows=sorted(frame.window_id.unique()),
                 fit_seconds=time.perf_counter()-start)
    return model

def diagnose(model,frame):
    raw=frame[FEATURES].to_numpy(float);w=weights(frame);y=frame.delta_target_c.to_numpy()
    z=(raw-model['mean'])/model['scale'];W=np.asarray(model['W']);v=np.asarray(model['v'])
    hidden=np.tanh(z@W+model['a']);pred=hidden@v+model['b']
    std=np.sqrt(w@((hidden-w@hidden)**2));mean=float(w@pred);intercept=float(w@y)
    within=[]
    for group in sorted(frame.window_id.unique()):
        part=pred[frame.window_id.eq(group).to_numpy()];within.append(float(np.std(part)))
    return dict(training_prediction_mse=float(w@((pred-y)**2)),penalty=float(model['lambda_value']*((W*W).sum()+v@v)),
                predicted_delta_mean_c=mean,predicted_delta_std_c=float(np.sqrt(w@((pred-mean)**2))),
                W_frobenius=float(np.linalg.norm(W)),v_l2=float(np.linalg.norm(v)),
                hidden_std_mean=float(np.mean(std)),hidden_std_max=float(np.max(std)),hidden_std_by_unit=std.tolist(),
                hidden_saturation_fraction=float(np.mean(w@(np.abs(hidden)>.99))),
                training_intercept_only_c=intercept,intercept_only_mse=float(w@((y-intercept)**2)),
                rms_difference_from_intercept_c=float(np.sqrt(w@((pred-intercept)**2))),
                within_window_prediction_std_min_c=min(within),within_window_prediction_std_max_c=max(within),
                within_window_prediction_std_mean_c=float(np.mean(within)))
