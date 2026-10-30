import numpy as np

def weighted_quantile(values,w,q):
    order=np.argsort(values,kind='stable');cumulative=np.cumsum(w[order]/w.sum())
    return float(values[order[min(np.searchsorted(cumulative,q,side='left'),len(order)-1)]])

def error_stats(error,baseline,correction,w):
    if len(error)==0:return {}
    w=w/w.sum();absolute=np.abs(error);before=np.abs(baseline);mean=float(w@correction)
    crossed=(baseline*error)<0
    result=dict(mae_c=float(w@absolute),rmse_c=float(np.sqrt(w@(error*error))),signed_mean_error_c=float(w@error),
                fraction_abs_error_above_1c=float(w@(absolute>1)),
                correction_mean_c=mean,correction_std_c=float(np.sqrt(w@((correction-mean)**2))),
                correction_min_c=float(np.min(correction)),correction_max_c=float(np.max(correction)),
                fraction_improved=float(w@(absolute<before)),fraction_worsened=float(w@(absolute>before)),fraction_tied=float(w@(absolute==before)),
                fraction_wrong_direction=float(w@((baseline*correction)>0)),fraction_crossing_target=float(w@crossed),
                fraction_harmful_overshoot=float(w@(crossed&(absolute>before))),
                fraction_zero_baseline_error=float(w@(baseline==0)),fraction_zero_correction=float(w@(correction==0)),fraction_exact_target=float(w@(error==0)))
    for label,q in [('median',.5),('p90',.9),('p95',.95),('p99',.99)]:result[label+'_absolute_error_c']=weighted_quantile(absolute,w,q)
    return result
