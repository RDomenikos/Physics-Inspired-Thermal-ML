"""Focused checks for outer/inner isolation, paired scoring and summary SD."""
import unittest
import run as r

class Checks(unittest.TestCase):
 def test_grouping_and_raw_overlap_boundaries(self):
  a=r.pd.read_csv(r.ROOT/'groups.csv')
  for ds,g in a.groupby('dataset'):
   self.assertEqual(g.group.nunique(),3)
   if ds.startswith('jetson'):
    for x in g.itertuples():
     for y in g.itertuples():
      if max(x.source_start,y.source_start)<=min(x.source_end,y.source_end):self.assertEqual(x.group,y.group)
   else:
    self.assertEqual(len(g),36);self.assertFalse(set(g.recording_id)&{'R03','R19','R21','R26'})
  for f in r.read(r.ROOT/'folds.json'):
   self.assertFalse(set(f['train_groups'])&set(f['test_groups']))
   self.assertFalse(set(f['train_recordings'])&set(f['test_recordings']))
 def test_fitted_state_and_prediction_checks(self):
  for name,file in [('intel','intel.parquet'),('jetson_inference','jetson.parquet')]:
   p=r.OUT/(name+'_models.json')
   if p.exists() and (r.ROOT/'input'/file).exists():
    r.verify(r.dataset(name,r.ROOT/'input'/file),r.pd.read_parquet(r.OUT/(name+'_predictions.parquet')),r.read(p))
 def test_seed_then_fold_summary(self):
  seeds=r.pd.read_csv(r.OUT/'seed_metrics.csv');fold=r.pd.read_csv(r.OUT/'fold_metrics.csv');summary=r.pd.read_csv(r.OUT/'summary.csv')
  for row in fold.itertuples():
   g=seeds.loc[seeds.dataset.eq(row.dataset)&seeds.fold.eq(row.fold)&seeds.method.eq(row.method)&seeds.recording_id.eq('ALL')]
   self.assertEqual(len(g),row.seeds_expected)
   if row.valid:
    for col in ['mae_c','rmse_c']:self.assertAlmostEqual(getattr(row,col),g[col].mean(),11)
  for row in summary.itertuples():
   g=fold.loc[fold.dataset.eq(row.dataset)&fold.method.eq(row.method)&fold.valid]
   self.assertEqual(row.valid_folds,len(g));self.assertEqual(row.planned_folds,3)
   for col in ['mae_c','rmse_c']:
    self.assertAlmostEqual(getattr(row,col+'_mean'),g[col].mean(),11)
    if len(g)>1:self.assertAlmostEqual(getattr(row,col+'_sd'),g[col].std(ddof=1),11)
 def test_paired_improvements(self):
  f=r.pd.read_csv(r.OUT/'fold_metrics.csv');p=r.pd.read_csv(r.OUT/'paired_folds.csv');s=r.pd.read_csv(r.OUT/'paired_summary.csv')
  for row in p.loc[p.valid].itertuples():
   g=f.loc[f.dataset.eq(row.dataset)&f.fold.eq(row.fold)].set_index('method')
   for col in ['mae_c','rmse_c']:
    d=g.loc[row.baseline,col]-g.loc[row.corrected,col]
    self.assertAlmostEqual(d,getattr(row,col+'_improvement'),11)
    self.assertAlmostEqual(100*d/g.loc[row.baseline,col],getattr(row,col+'_improvement_pct'),10)
  for row in s.itertuples():
   g=p.loc[p.dataset.eq(row.dataset)&p.corrected.eq(row.corrected)&p.valid]
   for col in ['mae_c','rmse_c']:
    self.assertAlmostEqual(getattr(row,col+'_improvement_mean'),g[col+'_improvement'].mean(),11)
    if len(g)>1:self.assertAlmostEqual(getattr(row,col+'_improvement_sd'),g[col+'_improvement'].std(ddof=1),11)
    self.assertEqual(sum(getattr(row,col+'_'+k) for k in ['improved','unchanged','worsened']),row.valid_folds)
if __name__=='__main__':unittest.main()
