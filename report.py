"""Report tables/figures from regenerated results, never from invented scores."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'report_outputs'

def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    jr=pd.read_csv(ROOT/'fixed/jetson/local/results/metrics.csv',dtype={'seed':str})
    ir=pd.read_csv(ROOT/'fixed/intel/local/results/metrics.csv',dtype={'seed':str})
    jr.to_csv(OUT/'fixed_jetson_all_metrics.csv',index=False);ir.to_csv(OUT/'fixed_intel_all_metrics.csv',index=False)
    pairs=[]
    for ds,table,aggregate,label,pairlist in [
        ('jetson',jr.loc[jr.stratum.eq('all')&jr.seed.eq('SEED_MEAN')], 'aggregate','label',[('sequential_base','sequential_thermal'),('H2','X_physics'),('H2','bounded')]),
        ('intel',ir.loc[ir.stratum.eq('all')&ir.seed.eq('SEED_MEAN')], 'EQUAL_RECORDING','recording_id',[('ridge_fixed','ridge_physics'),('H2','H2_physics'),('H2','H2_physics_bounded')])]:
        agg=table.loc[table.scope.eq(aggregate)] if ds=='jetson' else table.loc[table.recording_id.eq(aggregate)]
        rec=table.loc[table.scope.eq('recording')] if ds=='jetson' else table.loc[~table.recording_id.eq(aggregate)]
        for base,method in pairlist:
            b=agg.loc[agg.method.eq(base)].iloc[0];m=agg.loc[agg.method.eq(method)].iloc[0]
            row=dict(dataset=ds,scope='original_fixed_split',baseline=base,method=method)
            rb=rec.loc[rec.method.eq(base)].set_index(label);rm=rec.loc[rec.method.eq(method)].set_index(label).loc[rb.index]
            for metric in ['mae_c','rmse_c']:
                row['baseline_'+metric]=b[metric];row['method_'+metric]=m[metric];row[metric+'_improvement']=b[metric]-m[metric];row[metric+'_improvement_pct']=100*(b[metric]-m[metric])/b[metric]
                d=rb[metric]-rm[metric]
                for n,mask in [('wins',d>1e-10),('ties',abs(d)<=1e-10),('losses',d< -1e-10)]:row[metric+'_recording_'+n]=int(mask.sum())
            pairs.append(row)
    pairs=pd.DataFrame(pairs)
    aliases={'sequential_base':'Ridge','sequential_thermal':'Ridge_physics','ridge_fixed':'Ridge','ridge_physics':'Ridge_physics','X_physics':'H2_physics','bounded':'H2_bounded','H2_physics_bounded':'H2_bounded'}
    for col in ['baseline','method']:
        pairs['source_'+col]=pairs[col];pairs[col]=pairs[col].replace(aliases)
    pairs.to_csv(OUT/'fixed_comparisons.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for i,(ds,g) in enumerate(pairs.groupby('dataset')):
        for ax,metric in zip(axs[i],['mae_c','rmse_c']):
            x=np.arange(len(g));ax.bar(x-.18,g['baseline_'+metric],.36,label='Frozen baseline');ax.bar(x+.18,g['method_'+metric],.36,label='Correction');ax.set_xticks(x,g.method,rotation=15);ax.set_title(ds+' fixed split');ax.set_ylabel(metric+' (°C)');ax.legend(fontsize=8)
    save(fig,'fixed_comparisons')
    grouped=ROOT/'grouped/local/results'
    for name in ['seed_metrics','fold_metrics','summary','paired_folds','paired_summary']:
        pd.read_csv(grouped/(name+'.csv')).to_csv(OUT/('grouped_'+name+'.csv'),index=False)
    folds=pd.read_csv(grouped/'fold_metrics.csv');summary=pd.read_csv(grouped/'summary.csv')
    assert set(folds.dataset)=={'intel','jetson_inference','jetson_supplementary'}
    fig,axs=plt.subplots(3,2,figsize=(12,11))
    for i,(ds,g) in enumerate(summary.groupby('dataset')):
        methods=g.method.tolist();x=np.arange(len(g))
        for ax,metric in zip(axs[i],['mae_c','rmse_c']):
            ax.errorbar(x,g[metric+'_mean'],yerr=g[metric+'_sd'],fmt='ks',capsize=4,label='Mean ± fold sample SD')
            for k,(fold,f) in enumerate(folds.loc[folds.dataset.eq(ds)].groupby('fold')):
                ax.plot(x+(k-1)*.1,f.set_index('method').loc[methods,metric],'.',label=fold.replace('G_',''))
            ax.set_xticks(x,methods,rotation=20);ax.set_title(ds);ax.set_ylabel(metric+' (°C)');ax.legend(fontsize=7)
    save(fig,'grouped_folds')
    ref=ROOT/'refinement/local';m=pd.read_csv(ref/'metrics.csv');s=pd.read_csv(ref/'summary.csv');p=pd.read_csv(ref/'paired_summary.csv')
    aliases={'baseline':'Ridge','original':'Ridge_physics','removed':'Ridge_physics_no_intercept'}
    for name,table in [('metrics',m),('recordings',m.loc[m.level.eq('recording')]),('summary',s),('paired_summary',p)]:
        table=table.copy()
        for col in ['method','reference']:
            if col in table: table[col]=table[col].replace(aliases)
        table.to_csv(OUT/('refinement_'+name+'.csv'),index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4));methods=['baseline','original','removed']
    for ax,metric in zip(axs,['mae_c','rmse_c']):
        for fold,g in m.loc[m.level.eq('fold')].groupby('fold'):ax.plot(range(3),g.set_index('method').loc[methods,metric],'o-',label=fold.replace('G_',''))
        ax.set_xticks(range(3),['Baseline Ridge','Original correction','Intercept removed']);ax.set_ylabel(metric+' (°C)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'refinement_folds')
    fig,axs=plt.subplots(1,2,figsize=(11,5));g=m.loc[m.level.eq('fold')&m.method.ne('baseline')]
    for ax,metric in zip(axs,['mae_c','rmse_c']):
        for method,piece in g.groupby('method'):ax.plot(piece.fold.str.replace('G_','',regex=False),piece[metric+'_improvement_vs_baseline'],'o-',label=method)
        ax.axhline(0,color='gray');ax.tick_params(axis='x',rotation=20);ax.set_ylabel('Baseline − method '+metric+' (°C)');ax.legend()
    save(fig,'refinement_paired')
    fig,axs=plt.subplots(1,2,figsize=(12,5));r=m.loc[m.level.eq('recording')]
    for ax,metric in zip(axs,['mae_c','rmse_c']):
        for method,g in r.groupby('method'):ax.plot(g.recording_id,g[metric],'o-',label=method)
        ax.tick_params(axis='x',rotation=70);ax.set_ylabel(metric+' (°C)');ax.legend()
    save(fig,'refinement_recordings')
    row=p.loc[p.method.eq('removed')&p.reference.eq('baseline')].iloc[0]
    (OUT/'summary.md').write_text(f'''# Factual report-update material\n\nRetrospective intercept-removal refinement, evaluated on all three existing grouped folds without refitting. Saved alpha=1; only the correction offset is subtracted at prediction time. Baselines, slopes, preprocessing and groups remain unchanged. Original correction results remain beside it.\n\nMean fold MAE decreases {row.mae_c_mean_error_reduction_pct:.2f}% and mean fold RMSE {row.rmse_c_mean_error_reduction_pct:.2f}% relative to baseline. These are reductions of mean fold errors, not averages of fold percentages. Both metrics improve in two of three folds; late October remains slightly worse than baseline. The variant was introduced after inspecting the original grouped failure. It has not been independently confirmed or evaluated on Intel or neural models.\n\nOriginal fixed-split evidence, inference-only grouped Jetson, Intel leave-one-date-out, and supplementary Jetson inference/training workloads are separate scopes. Never pool hardware errors. Neural seeds are scored separately, then averaged within each fold; SD across three folds uses ddof=1. Seed SD columns are separate. Positive improvement means baseline minus method error. Historical fixed tables use method-minus-comparator columns; their sign is opposite. Refinement frozen-reference change columns also use method minus reference. New comparison CSVs use positive improvement consistently.\n\nAll folds, seeds and adverse outcomes are retained. Supplementary grouped aggregates are reused, not newly refitted; their raw preparation route is not supplied. Original Jetson neural correction worsens fixed-split MAE despite improving RMSE. Generalization and physical-mechanism claims remain limited. No energy savings are measured.\n\nCSV files preserve per-recording and seed detail; PDF/PNG figures show fixed comparisons, all grouped fold points with mean ± sample SD, refinement fold and recording comparisons, and paired changes. Wins/ties/losses are in paired-summary tables and fixed_comparisons.csv. Baseline ties are numerical at 1e-10°C. These are descriptive results on previously inspected recordings, not independent replicates or confidence intervals.\n''')
    print('Report outputs generated from replayed metrics; supplementary evidence separately labelled.')
if __name__=='__main__':main()
