import numpy as np
ENRICHMENT_FAMILIES=('physics_enriched','generic_enriched')

def weights(frame):
    counts = frame.groupby('window_id').window_id.transform('size').to_numpy()
    return 1.0 / (frame.window_id.nunique() * counts)

def fit_ridge(frame, features, target, config):
    x = frame[features].to_numpy(float)
    w = weights(frame)
    mean = w @ x
    std = np.sqrt(w @ ((x - mean) ** 2))
    scale = np.where(std < config['constant_std_threshold'], 1.0, std)
    z = (x - mean) / scale
    # Explicit intercept makes the specified objective independent of roundoff in centering.
    design = np.column_stack([np.ones(len(frame)), z])
    penalty = np.diag([0.0] + [config['ridge_lambda']] * len(features))
    beta = np.linalg.solve(design.T @ (w[:, None] * design) + penalty,
                           design.T @ (w * np.asarray(target)))
    return dict(features=features, feature_mean=mean.tolist(), feature_std=std.tolist(),
                feature_scale=scale.tolist(), coefficients_standardized=beta[1:].tolist(),
                intercept_delta_c=float(beta[0]), lambda_value=config['ridge_lambda'])

def ridge_delta(model, frame):
    z = (frame[model['features']].to_numpy(float) - model['feature_mean']) / model['feature_scale']
    return model['intercept_delta_c'] + z @ model['coefficients_standardized']

def radiation_feature(theta, reference_k, scale_c):
    return (6*reference_k**2*theta**2 + 4*reference_k*theta**3 + theta**4) / (4*reference_k**3*scale_c)

def enrichment_inputs(model, frame):
    z = (frame[model['features']].to_numpy(float)-model['feature_mean'])/model['feature_scale']
    coordinates = model['coordinates']
    theta = frame.cpu_current_c.to_numpy()+273.15-coordinates['temperature_reference_k']
    x = theta/coordinates['temperature_scale_k']
    power = (frame.power_VDD_CPU_CV_mean_mw.to_numpy()/1000-coordinates['power_mean_w'])/coordinates['power_scale_w']
    if model['family'] == 'physics_enriched':
        raw = np.column_stack([x*np.abs(x)**0.25,
                               radiation_feature(theta, coordinates['temperature_reference_k'], coordinates['temperature_scale_k']),
                               power*x])
    else:
        raw = np.column_stack([x**2, x**3, power*x])
    return np.column_stack([np.ones(len(frame)),z]), raw

def feature_geometry(values, w):
    if values.shape[1] == 0:
        return dict(singular_values=[],condition_number=None,rank=0,correlation=[])
    weighted = np.sqrt(w)[:,None]*values
    singular = np.linalg.svd(weighted,compute_uv=False)
    centered = values-w@values
    cov = centered.T@(w[:,None]*centered)
    norm = np.sqrt(np.diag(cov))
    corr = cov/(norm[:,None]*norm[None,:])
    return dict(singular_values=singular.tolist(),condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                rank=int(np.linalg.matrix_rank(weighted)),correlation=corr.tolist())

def fit_centered_physics(frame, config, family='physics_enriched'):
    """Original coordinates and three fixed features; centering/scaling, never projection."""
    w=weights(frame)
    features=config['temperature_features']+config['telemetry_extra_features']
    x=frame[features].to_numpy(float); mean=w@x; std=np.sqrt(w@((x-mean)**2))
    temperature=frame.cpu_current_c.to_numpy()+273.15
    power=frame.power_VDD_CPU_CV_mean_mw.to_numpy()/1000
    tm=float(w@temperature); ts=float(np.sqrt(w@((temperature-tm)**2)))
    pm=float(w@power); ps=float(np.sqrt(w@((power-pm)**2)))
    assert family in ENRICHMENT_FAMILIES
    model=dict(family=family,features=features,feature_mean=mean.tolist(),feature_std=std.tolist(),
               feature_scale=np.where(std<1e-12,1,std).tolist(),
               coordinates=dict(temperature_reference_k=tm,temperature_std_k=ts,temperature_scale_k=ts if ts>1e-12 else 1,
                                power_mean_w=pm,power_std_w=ps,power_scale_w=ps if ps>1e-12 else 1))
    _,raw=enrichment_inputs(model,frame)
    mu=w@raw; sd=np.sqrt(w@((raw-mu)**2)); threshold=np.maximum(1e-12,1e-10*np.sqrt(w@(raw**2)))
    kept=sd>threshold
    model.update(thermal_mean=mu.tolist(),thermal_std=sd.tolist(),thermal_scale=np.where(kept,sd,1).tolist(),
                 kept_features=kept.tolist(),degeneracy_threshold=threshold.tolist(),transform='center and standardize only; no projection')
    return model

def centered_design(model, frame):
    original,raw=enrichment_inputs(model,frame)
    thermal=(raw-model['thermal_mean'])/model['thermal_scale']
    thermal[:,~np.asarray(model['kept_features'],dtype=bool)]=0
    return original,thermal

def fit_controlled_ridge(frame,config,regularization,target=None,correction=False,enabled=True,family='physics_enriched'):
    model=fit_centered_physics(frame,config,family)
    original,thermal=centered_design(model,frame)
    kept=np.asarray(model['kept_features'],dtype=bool) if enabled else np.zeros(3,dtype=bool)
    design=np.column_stack([np.ones(len(frame)) if correction else original,thermal[:,kept]])
    y=frame.delta_target_c.to_numpy() if target is None else np.asarray(target)
    w=weights(frame); penalty=np.r_[0,np.full(design.shape[1]-1,regularization)]
    augmented=np.vstack([np.sqrt(w)[:,None]*design,np.diag(np.sqrt(penalty))])
    coefficients=np.linalg.lstsq(augmented,np.r_[np.sqrt(w)*y,np.zeros(design.shape[1])],rcond=1e-12)[0]
    offset=1 if correction else 12
    gamma=np.zeros(3); gamma[kept]=coefficients[offset:]
    model.update(lambda_value=regularization,correction=correction,extension_enabled=enabled,
                 intercept_delta_c=float(coefficients[0]),gamma=gamma.tolist(),
                 coefficients_standardized=[] if correction else coefficients[1:12].tolist(),
                 training_windows=sorted(frame.window_id.unique().tolist()),training_n=len(frame),
                 objective=float(w@((design@coefficients-y)**2)+regularization*(coefficients[1:]@coefficients[1:])),
                 thermal_geometry=feature_geometry(thermal[:,kept],w))
    return model

def controlled_delta(model,frame):
    original,thermal=centered_design(model,frame)
    value=np.full(len(frame),model['intercept_delta_c'])
    if not model['correction']:
        value+=original[:,1:]@np.asarray(model['coefficients_standardized'])
    return value+thermal@np.asarray(model['gamma'])
