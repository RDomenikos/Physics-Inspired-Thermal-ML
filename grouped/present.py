"""Aggregate-only figures and factual summary from saved fold tables; no fitting."""
import os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'jetson_inference':'Jetson — ten inference windows','intel':'Intel i7-12700 — 36 valid recordings','jetson_supplementary':'Jetson supplementary — fifteen inference/training windows'}
ORDER=['jetson_inference','intel','jetson_supplementary']
COLORS={'Ridge_physics':'#c27826','H2_physics':'#247b83','H2_bounded':'#67994a'}
NAMES={'Ridge':'Ridge','Ridge_physics':'Ridge + physics','H2':'H2','H2_physics':'H2 + physics','H2_bounded':'H2 + bounded physics'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=ROOT/'results');parser.add_argument('--figures-dir',type=Path,default=ROOT/'figures');args=parser.parse_args()
 out=args.results_dir;figdir=args.figures_dir;figdir.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv(out/'fold_metrics.csv');s=pd.read_csv(out/'summary.csv');p=pd.read_csv(out/'paired_folds.csv');ps=pd.read_csv(out/'paired_summary.csv')
 datasets=[d for d in ORDER if d in set(f.dataset)]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def finish(fig,name):
  fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');fig.savefig(figdir/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
 def axes():return plt.subplots(len(datasets),2,figsize=(12,3.6*len(datasets)),squeeze=False)
 fig,axs=axes()
 for i,ds in enumerate(datasets):
  pairs=p.loc[p.dataset.eq(ds)];values=f.loc[f.dataset.eq(ds)].set_index(['fold','method'])
  for j,metric in enumerate(['mae_c','rmse_c']):
   ax=axs[i,j];labels=[]
   for k,r in enumerate(pairs.itertuples()):
    labels.append(r.fold.replace('G_','')+'\n'+NAMES[r.corrected].replace(' + ','+'))
    if r.valid:ax.plot([k-.12,k+.12],[values.loc[(r.fold,r.baseline),metric],values.loc[(r.fold,r.corrected),metric]],'-o',color=COLORS[r.corrected],ms=4)
   ax.set_xticks(range(len(labels)),labels,rotation=60,ha='right');ax.set_ylabel(metric.removesuffix('_c').upper()+' (°C)');ax.set_title(LABELS[ds]);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Outer-fold paired values: baseline (left) → correction (right)');fig.text(.5,.005,'Neural points are means of five separately scored seeds; groups/scopes are never pooled.',ha='center');finish(fig,'01_fold_pairs')
 fig,axs=axes()
 for i,ds in enumerate(datasets):
  a=s.loc[s.dataset.eq(ds)];methods=[m for m in NAMES if m in set(a.method)];a=a.set_index('method')
  for j,metric in enumerate(['mae_c','rmse_c']):
   ax=axs[i,j]
   for k,m in enumerate(methods):
    r=a.loc[m];ax.errorbar(k,r[metric+'_mean'],yerr=r[metric+'_sd'],fmt='o',color='#29465b',capsize=5)
    points=f.loc[f.dataset.eq(ds)&f.method.eq(m)&f.valid,metric];ax.scatter(k+np.linspace(-.12,.12,len(points)),points,s=20,facecolors='none',edgecolors='#9b5b37')
   ax.set_xticks(range(len(methods)),[NAMES[m] for m in methods],rotation=25,ha='right');ax.set_title(LABELS[ds]);ax.set_ylabel('Mean fold '+metric.removesuffix('_c').upper()+' (°C)');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Arithmetic fold means ± sample SD, with all valid fold points');fig.text(.5,.005,'Descriptive SD across three folds, not a confidence interval or seed SD. Mean fold RMSE is not pooled RMSE.',ha='center');finish(fig,'02_mean_sd')
 fig,axs=axes()
 for i,ds in enumerate(datasets):
  a=p.loc[p.dataset.eq(ds)];folds=sorted(a.fold.unique())
  for j,metric in enumerate(['mae_c','rmse_c']):
   ax=axs[i,j];ax.axhline(0,color='black',lw=.8)
   for method,g in a.groupby('corrected'):
    ax.plot([folds.index(x) for x in g.fold],g[metric+'_improvement'],'o-',color=COLORS[method],label=NAMES[method])
   ax.set_xticks(range(len(folds)),[x.replace('G_','') for x in folds],rotation=15);ax.set_title(LABELS[ds]);ax.set_ylabel('Baseline − corrected '+metric.removesuffix('_c').upper()+' (°C)');ax.legend(fontsize=7)
 fig.suptitle('Paired improvement per outer fold — positive values favor physics');finish(fig,'03_paired_improvement')
 fixed=pd.read_csv(ROOT/'reference/fixed_split.csv');dslist=[d for d in datasets if d in set(fixed.dataset)]
 fig,axs=plt.subplots(len(dslist),2,figsize=(12,3.7*len(dslist)),squeeze=False)
 for i,ds in enumerate(dslist):
  a=s.loc[s.dataset.eq(ds)].set_index('method');b=fixed.loc[fixed.dataset.eq(ds)].set_index('method');methods=[m for m in NAMES if m in a.index and m in b.index]
  for j,metric in enumerate(['mae_c','rmse_c']):
   ax=axs[i,j];x=np.arange(len(methods));ax.plot(x-.08,b.loc[methods,metric],'s',label='Original fixed split');ax.errorbar(x+.08,a.loc[methods,metric+'_mean'],yerr=a.loc[methods,metric+'_sd'],fmt='o',capsize=4,label='Grouped fold mean ± SD')
   ax.set_xticks(x,[NAMES[m] for m in methods],rotation=25,ha='right');ax.set_title(LABELS[ds]);ax.set_ylabel(metric.removesuffix('_c').upper()+' (°C)');ax.legend(fontsize=8)
 fig.suptitle('Different evaluation designs: original fixed split versus grouped robustness');fig.text(.5,.005,'Different train/test compositions and estimands; this is not a paired test of improvement between evaluation designs.',ha='center');finish(fig,'04_original_vs_grouped')
 lines=['# Grouped evaluation findings','','Three-fold grouped cross-validation with previously selected, fixed hyperparameters. This is retrospective robustness assessment, not fully nested model-selection validation. Every fold refits its models. Neural summaries average seeds within folds first, then folds equally. SD uses ddof=1. No pooled OOF metric is substituted.','', '| Scope / method | MAE mean ± SD (°C) | Mean fold RMSE ± SD (°C) | Valid/planned folds |','|---|---:|---:|---:|']
 for ds in datasets:
  for r in s.loc[s.dataset.eq(ds)].itertuples():lines.append(f'| {LABELS[ds]} / {NAMES[r.method]} | {r.mae_c_mean:.6f} ± {r.mae_c_sd:.6f} | {r.rmse_c_mean:.6f} ± {r.rmse_c_sd:.6f} | {r.valid_folds}/3 |')
 lines+=['','Paired improvements use baseline minus corrected error. Positive is better; no favorable folds or seeds were selected.','', '| Scope / correction | MAE improvement mean ± SD (°C) | RMSE improvement mean ± SD (°C) | MAE wins / ties / losses | RMSE wins / ties / losses |','|---|---:|---:|---:|---:|']
 for r in ps.itertuples():lines.append(f'| {LABELS[r.dataset]} / {NAMES[r.corrected]} | {r.mae_c_improvement_mean:.6f} ± {r.mae_c_improvement_sd:.6f} | {r.rmse_c_improvement_mean:.6f} ± {r.rmse_c_improvement_sd:.6f} | {r.mae_c_improved}/{r.mae_c_unchanged}/{r.mae_c_worsened} | {r.rmse_c_improved}/{r.rmse_c_unchanged}/{r.rmse_c_worsened} |')
 lines+=['','Percent improvements and their sample SD are in paired_summary.csv; each fold percentage has its own baseline denominator.','', 'Jetson inference has only 39 origins in the 23 October group, versus 2,675 on 22 October and 1,457 in the linked late-October group. These are heterogeneous, small dependence blocks, not independent trials. Intel date folds also change workload composition. Earlier held-out dates trained on later dates are not prospective forecasts.','', 'The supplementary fifteen-window Jetson results are reused and independently checked, with no new fits. Their scope includes training workloads and remains separate. The original fixed-split comparison figure uses different training/evaluation compositions; it is not an additional paired improvement estimate.','', 'The original reports, models, results and publication folder are unchanged. Refer to failure records for any unresolved fits or infeasible procedures. No confidence intervals based on rows/seeds are claimed.']
 if 'jetson_inference' in datasets:
  lines+=['', '## Jetson inference: all three folds', '', 'H2 values are means across five separately scored seeds. All three folds are valid. The late-October fold trains on the original development recordings, but its corrections use whole-group OOF residuals rather than the original recording-window residual procedure; equal receiver scores do not imply an identical correction.', '', '| Held-out group | Method | MAE (°C) | Balanced RMSE (°C) |', '|---|---|---:|---:|']
  for row in f.loc[f.dataset.eq('jetson_inference')].itertuples():
   lines.append(f'| {row.fold} | {NAMES[row.method]} | {row.mae_c:.6f} | {row.rmse_c:.6f} |')
  lines+=['', '| Held-out group | Correction | MAE improvement (°C / %) | RMSE improvement (°C / %) |', '|---|---|---:|---:|']
  for row in p.loc[p.dataset.eq('jetson_inference')].itertuples():
   lines.append(f'| {row.fold} | {NAMES[row.corrected]} | {row.mae_c_improvement:.6f} / {row.mae_c_improvement_pct:.3f}% | {row.rmse_c_improvement:.6f} / {row.rmse_c_improvement_pct:.3f}% |')
  lines+=['', 'In inference-only Jetson CV, Ridge correction improves mean MAE slightly but worsens mean RMSE; it improves both metrics in two folds and worsens both in the late-October fold. Both H2 corrections worsen mean MAE and mean RMSE, improving both only in the 23 October fold (39 origins). Bounding reduces the average deterioration relative to unbounded H2 correction. These outcomes are preserved without retuning.']
 (out/'findings.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
