import unittest
import run as r
class ComparisonChecks(unittest.TestCase):
 def test_reference_and_metrics(self):
  a=r.read(r.MODEL);p=r.pd.read_csv(r.OUT/'predictions.csv',float_precision='round_trip');m=r.pd.read_csv(r.OUT/'metrics.csv');m['seed']=m.seed.map(lambda s:s if str(s).startswith('SEED_') else int(s));r.verify(p,m,a)
 def test_only_requested_methods_and_complete_origins(self):
  p=r.pd.read_csv(r.OUT/'predictions.csv');self.assertEqual(set(p.method),{'ridge_fixed','ridge_physics','H2','H2_physics','H2_physics_bounded'});self.assertEqual(p.recording_id.nunique(),7)
  for _,g in p.groupby(['method','seed']):self.assertEqual(len(g),6035)
  r.np.testing.assert_allclose(p.predicted_delta_c,p.baseline_delta_c+p.applied_correction_c,atol=1e-12,rtol=0)
 def test_training_exclusion_and_bounds(self):
  a=r.read(r.MODEL);f=r.pd.read_parquet(r.ROOT/'local/features.parquet');dev=f.loc[f.split.eq('development')]
  for key,m in a['models'].items():
   self.assertEqual(set(m['training_groups']),{'2025-09-29','2025-09-30'});c=a['corrections'][key+'_physics'];delta=(1 if key=='ridge_fixed' else .25)*(c['c']+r.design(c,dev)@c['gamma']);q=r.quantile(abs(delta),r.weights(dev),.95);self.assertAlmostEqual(a['bounds'][key+'_physics'],q if q else 1e-12,12)
if __name__=='__main__':unittest.main()
