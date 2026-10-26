"""Descriptive window statistics and timestamp-matched changes; no predictions."""
import json
import numpy as np
import pandas as pd
from prepare_stage2 import ROOT, DERIVED, REPORT, TEMPERATURES, RAILS, dump

HORIZONS=(10,30,60)
MATCH_TOLERANCE_S=0.0  # Exact matching for one-second-resolution raw timestamps.


def distribution(values):
    arr=np.asarray(values,dtype=float)
    arr=arr[np.isfinite(arr)]
    if not len(arr):
        return dict(count=0)
    q=np.quantile(arr,[0,.05,.25,.5,.75,.95,1])
    return dict(count=int(len(arr)),minimum=float(q[0]),p05=float(q[1]),p25=float(q[2]),
                median=float(q[3]),p75=float(q[4]),p95=float(q[5]),maximum=float(q[6]),
                mean=float(arr.mean()),std_population=float(arr.std()),range=float(q[6]-q[0]))


def horizon_pairs(seconds, values, horizon):
    """Exact target timestamps only, with no interpolation or nearest-row substitution."""
    seconds=np.asarray(seconds,dtype=float)
    values=np.asarray(values,dtype=float)
    lookup={float(t):i for i,t in enumerate(seconds)}
    changes=[];achieved=[]
    unavailable=dict(outside_window=0,no_exact_target_timestamp=0,missing_temperature=0)
    for i,t in enumerate(seconds):
        target=t+horizon
        if target>seconds[-1]:
            unavailable['outside_window']+=1
        elif target not in lookup:
            unavailable['no_exact_target_timestamp']+=1
        elif not (np.isfinite(values[i]) and np.isfinite(values[lookup[target]])):
            unavailable['missing_temperature']+=1
        else:
            j=lookup[target]
            changes.append(values[j]-values[i]);achieved.append(seconds[j]-t)
    return dict(requested_horizon_s=horizon,timestamp_match_tolerance_s=MATCH_TOLERANCE_S,
                potential_origins=len(seconds),available_pairs=len(changes),unavailable_pairs=sum(unavailable.values()),
                unavailable_reasons=unavailable,achieved_horizon_s=distribution(achieved),
                temperature_change_c=distribution(changes),
                interpretation='Endpoint differences only; intervening gaps are not interpolated. Overlapping pairs are autocorrelated.')


def selected_frame(window, context=False):
    # Predicate pushdown limits materialization even for multi-day logs.
    import pyarrow.parquet as pq
    if context:
        start=pd.Timestamp(window['wall_start']).to_pydatetime()-pd.Timedelta(seconds=60)
        end=pd.Timestamp(window['wall_end']).to_pydatetime()+pd.Timedelta(seconds=60)
        filters=[('wall_time','>=',start),('wall_time','<=',end)]
    else:
        filters=[('raw_sample_index','>=',window['raw_start_index']),('raw_sample_index','<=',window['raw_end_index'])]
    return pq.read_table(DERIVED/'canonical'/f"{window['recording_id']}.parquet",filters=filters).to_pandas()


def diagnose(window):
    frame=selected_frame(window)
    seconds=(frame.wall_time-frame.wall_time.iloc[0]).dt.total_seconds().to_numpy()
    temps={}
    for name in TEMPERATURES:
        values=frame[f'temperature_{name}_c'].to_numpy()
        stats=distribution(values)
        stats.update(initial_c=float(values[0]) if np.isfinite(values[0]) else None,
                     final_c=float(values[-1]) if np.isfinite(values[-1]) else None,
                     net_change_c=float(values[-1]-values[0]) if np.isfinite(values[-1]-values[0]) else None,
                     missing_samples=int(np.isnan(values).sum()))
        if name in ('cpu','soc0','soc1','soc2'):
            stats['horizon_changes']={str(h):horizon_pairs(seconds,values,h) for h in HORIZONS}
        temps[name]=stats
    active_cols=[f'cpu{i}_active' for i in range(12)]
    active=frame[active_cols].sum(axis=1,min_count=12)
    load=frame[[f'cpu{i}_utilization_pct' for i in range(12)]].mean(axis=1)
    clock=frame[[f'cpu{i}_clock_mhz' for i in range(12)]]
    variability={f'power_{rail}_instantaneous_mw':distribution(frame[f'power_{rail}_instantaneous_mw']) for rail in RAILS}
    variability.update(cpu_mean_active_utilization_pct=distribution(load),gpu_utilization_pct=distribution(frame.gpu_utilization_pct),
                       active_core_count=distribution(active))
    transitions={f'cpu{i}':int(((frame[f'cpu{i}_clock_mhz'].diff()!=0)&frame[f'cpu{i}_clock_mhz'].notna()&frame[f'cpu{i}_clock_mhz'].shift().notna()).sum()) for i in range(12)}
    active_changes=int((frame[active_cols].ne(frame[active_cols].shift()).any(axis=1).iloc[1:]).sum())
    internal=np.diff(seconds)
    numeric=[c for c in frame.columns if c.startswith(('temperature_','power_','cpu','gpu_','ram_'))]
    return dict(window_id=window['window_id'],recording_id=window['recording_id'],
                category=window['category'],status=window['inclusion_status'],validation_group=window['proposed_validation_group'],
                sample_count=len(frame),valid_cpu_samples=int(frame.temperature_cpu_c.notna().sum()),duration_s=float(seconds[-1]),
                temperatures=temps,variability=variability,
                timing=dict(gap_count=int((internal>1).sum()),max_interval_s=float(internal.max()),
                            interval_counts={str(k):int(v) for k,v in zip(*np.unique(internal,return_counts=True))}),
                missing_counts={c:int(frame[c].isna().sum()) for c in numeric},
                entirely_missing_channels=[c for c in numeric if frame[c].isna().all()],
                visible_configuration=dict(active_mask_transitions=active_changes,per_core_clock_transitions=transitions,
                                           min_active_clock_mhz=float(clock.min().min()),max_active_clock_mhz=float(clock.max().max()),
                                           interpretation='Clock changes indicate observed frequency variation, not proven power-mode transitions or thermal throttling.'),
                overlap_selected=window['selected_window_overlaps'],overlap_raw=window['full_raw_period_overlaps'])


def main():
    windows=json.loads((ROOT/'manifests/windows.json').read_text())
    summaries=[diagnose(w) for w in windows]
    dump(REPORT/'window-diagnostics.json',summaries)
    rows=[];horizon_rows=[]
    for s in summaries:
        row={k:s[k] for k in ('window_id','category','status','validation_group','sample_count','valid_cpu_samples','duration_s')}
        row.update(gap_count=s['timing']['gap_count'],max_interval_s=s['timing']['max_interval_s'],
                   active_mask_transitions=s['visible_configuration']['active_mask_transitions'])
        for name in ('cpu','soc0','soc1','soc2'):
            stats=s['temperatures'][name]
            for k in ('initial_c','final_c','minimum','maximum','net_change_c','range'):
                row[f'{name}_{k}']=stats.get(k)
            for h,d in stats['horizon_changes'].items():
                hr=dict(window_id=s['window_id'],sensor=name,horizon_s=int(h),tolerance_s=MATCH_TOLERANCE_S,
                        available_pairs=d['available_pairs'],unavailable_pairs=d['unavailable_pairs'])
                hr.update(d['unavailable_reasons'])
                hr.update({f'delta_c_{k}':v for k,v in d['temperature_change_c'].items()})
                hr.update({f'achieved_s_{k}':v for k,v in d['achieved_horizon_s'].items()})
                horizon_rows.append(hr)
        for channel,d in s['variability'].items():
            for k in ('minimum','maximum','mean','std_population','range'):
                row[f'{channel}_{k}']=d.get(k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT/'window-summary.csv',index=False)
    pd.DataFrame(horizon_rows).to_csv(REPORT/'horizon-changes.csv',index=False)
    lines=['# Author-selected window summary','',
           'Temperature is °C; time is actual elapsed seconds. Full sensor statistics and variability: window-summary.csv and window-diagnostics.json. No predictive performance is calculated.','',
           '| Window | Status | Group (wall date) | N | Duration s | CPU initial | CPU final | CPU range | CPU net Δ | Gaps |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        t=s['temperatures']['cpu']
        lines.append(f"| {s['recording_id']} | {s['status']} | {s['validation_group'].replace('wall_date_','')} | {s['sample_count']} | {s['duration_s']:.0f} | {t['initial_c']:.3f} | {t['final_c']:.3f} | {t['range']:.3f} | {t['net_change_c']:+.3f} | {s['timing']['gap_count']} |")
    (REPORT/'window-summary.md').write_text('\n'.join(lines)+'\n')
    candidates=[w for w in windows if w['inclusion_status']=='candidate']
    groups={g:[w['recording_id'] for w in candidates if w['proposed_validation_group']==g]
            for g in sorted({w['proposed_validation_group'] for w in candidates})}
    # Embargo complete dates that contain a held-out workload/mode.
    feasibility={}
    for field in ('documented_workload','nominal_mode'):
        options=[]
        for value in sorted({w[field] for w in candidates}):
            test_groups=sorted({w['proposed_validation_group'] for w in candidates if w[field]==value})
            train_groups=sorted(set(groups)-set(test_groups))
            options.append(dict(value=value,test_groups=test_groups,remaining_development_groups=train_groups,
                                target_test_windows=[w['recording_id'] for w in candidates if w[field]==value],
                                development_windows=[w['recording_id'] for w in candidates if w['proposed_validation_group'] in train_groups],
                                nonempty_split=bool(train_groups),
                                limitation='Whole dates embargoed; tiny/confounded coverage even if nonempty. Other workloads/modes on test dates are not development data.'))
        feasibility[field]=options
    dump(REPORT/'evaluation-proposal.json',dict(candidate_count=len(candidates),groups=groups,
         grouping='Conservative raw-wall-date assumption, not verified sessions; all selected overlaps embargoed together.',
         proposed_development_groups=['wall_date_2024-10-22','wall_date_2024-10-23'],
         proposed_final_holdout_groups=['wall_date_2024-10-25'],
         alternative='Leave-one-date-out exploratory stress test; do not call it independent confirmation after development use.',
         fully_blind=False,workload_mode_feasibility=feasibility,
         allowed_inputs='Only past/current measurements and causally computed history at forecast origin.',
         prohibited_inputs=['measured future power','future temperatures','centred smoothing','whole-run normalization'],
         models_fitted=False))
    print((REPORT/'window-summary.md').read_text())


if __name__=='__main__':
    main()
