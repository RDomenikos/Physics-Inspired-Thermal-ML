"""Focused checks for raw-source preparation and saved-model replay portability."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import run as launcher

ROOT = Path(__file__).resolve().parent


@unittest.skipUnless((ROOT/'jetson/local/features.parquet').exists(), 'Run the Jetson raw-source workflow first')
class JetsonChecks(unittest.TestCase):
    def setUp(self):
        self.f = pd.read_parquet(ROOT/'jetson/local/features.parquet')
        self.p = pd.read_parquet(ROOT/'jetson/local/results/predictions.parquet')
        self.core = launcher.load('replay_checks', ROOT/'jetson/run.py')

    def test_counts_timestamps_and_rows(self):
        self.assertEqual(self.f.groupby('split').size().to_dict(), {'development': 2714, 'heldout': 1457})
        self.assertFalse(self.f.duplicated(['recording_id','origin_timestamp']).any())
        delta = pd.to_datetime(self.f.target_timestamp)-pd.to_datetime(self.f.origin_timestamp)
        self.assertTrue(delta.dt.total_seconds().eq(60).all())
        self.assertTrue(self.f.max_history_interval_s.le(2).all())
        for _, group in self.p.groupby(['experiment_id','method','seed']):
            self.assertEqual(len(group),1457)
        np.testing.assert_allclose(self.p.predicted_temperature_c,
            self.p.baseline_prediction_c+self.p.applied_physics_correction_c,atol=1e-12,rtol=0)

    def test_aggregate_reference(self):
        keys=['experiment_id','split','seed','method','scope','label','stratum']
        expected=pd.read_csv(ROOT/'jetson/results/reference/metrics.csv',dtype={'seed':str})
        got=pd.read_csv(ROOT/'jetson/local/results/metrics.csv',dtype={'seed':str})
        j=expected.merge(got,on=keys,suffixes=('_old','_new'),validate='one_to_one')
        self.assertEqual(len(j),len(expected))
        for name in ['mae_c','rmse_c']:
            np.testing.assert_allclose(j[name+'_old'],j[name+'_new'],atol=1e-10,rtol=0,equal_nan=True)

    def test_saved_scaling_is_development_only(self):
        cfg=launcher.read(ROOT/'jetson/manifests/config.json')
        names=cfg['temperature_features']+cfg['telemetry_extra_features']
        train=self.f.loc[self.f.split.eq('development')]
        self.assertEqual(set(train.recording_id),set(cfg['development_windows']))
        self.assertTrue(set(train.group).isdisjoint(set(self.f.loc[self.f.split.eq('heldout'),'group'])))
        w=self.core.weights(train);raw=train[names].to_numpy();mu=w@raw
        sd=np.sqrt(w@((raw-mu)**2))
        registry=launcher.read(ROOT/'jetson/models/registry.json')
        model=next(v['baseline'] for k,v in registry.items() if k.startswith('neural_refined/') and k.endswith('/0/H2'))
        np.testing.assert_allclose(model['mean'],mu,atol=1e-10,rtol=0)
        np.testing.assert_allclose(model['scale'],sd,atol=1e-10,rtol=0)

    def test_future_does_not_change_features(self):
        # Reuses the original Stage 3 causality test's perturb-future-observations check.
        sys.path.insert(0,str(ROOT/'jetson/scripts'))
        import stage3_core as c
        f=pd.read_parquet(ROOT/'jetson/local/derived/canonical/depth_15W.parquet')
        data=c.arrays(f);cfg=launcher.read(ROOT/'jetson/manifests/config.json')
        origin=100
        before,reason,_=c.causal_features(data,origin,cfg)
        self.assertIsNone(reason)
        altered={k:v.copy() for k,v in data.items()}
        for key in ['cpu','power','gpu','loads']:
            altered[key][origin+1:]=999999
        after,reason,_=c.causal_features(altered,origin,cfg)
        self.assertIsNone(reason)
        np.testing.assert_array_equal(before,after)


if __name__=='__main__':
    unittest.main()
