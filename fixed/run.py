"""Thin launcher for original preparation routines and frozen-model evaluation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def stage_file(source, destination, entry, jetson=False):
    if not source.is_file():
        raise SystemExit('Missing source recording: ' + str(source) + '\nSee README.md for the pinned publisher download.')
    b = source.read_bytes()
    assert len(b) == entry['expected_bytes' if jetson else 'bytes'], source.name
    assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == entry['git_blob_sha1'], source.name
    if jetson:
        assert hashlib.sha256(b).hexdigest() == entry['sha256'], source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare_jetson(source):
    import pandas as pd
    base = ROOT / 'jetson'
    local = base / 'local'
    local.mkdir(exist_ok=True)
    for item in read(base / 'manifests/sources.json'):
        path = Path(item['local_path'])
        stage_file(source / Path(*path.parts[2:]), local / path, item, True)
    sys.path.insert(0, str(base / 'scripts'))
    import prepare_stage2 as prep
    import diagnose_stage2 as diagnose
    import stage3_core as core
    # Only IO roots change. Parser, selected windows and causal feature equations are original.
    prep.ROOT = local
    prep.DERIVED = local / 'derived'
    diagnose.DERIVED = prep.DERIVED
    cfg = read(base / 'manifests/config.json')
    cfg['horizons_s'] = [60]  # Select the report's existing horizon; do not alter its extraction.
    sources = {x['local_path']: x for x in read(base / 'manifests/sources.json')}
    rows = []
    for record in read(base / 'manifests/recordings.json'):
        print('Prepare Jetson:', record['recording_id'], flush=True)
        window, _ = prep.prepare_record(record, sources[record['raw_path']])
        frame = diagnose.selected_frame(window)
        got, _ = core.eligible_examples(window, core.arrays(frame), cfg)
        rows.extend(got)
    features = pd.DataFrame(rows)
    assert features.groupby('split').size().to_dict() == {'development': 2714, 'heldout': 1457}
    assert not features.duplicated(['recording_id', 'origin_timestamp']).any()
    assert ((pd.to_datetime(features.target_timestamp) - pd.to_datetime(features.origin_timestamp)).dt.total_seconds() == 60).all()
    features.to_parquet(local / 'features.parquet', index=False)
    return features


def evaluate_jetson(features):
    import numpy as np
    import pandas as pd
    base = ROOT / 'jetson'
    core = load('jetson_replay', base / 'run.py')
    cfg = read(base / 'manifests/config.json')
    names = cfg['temperature_features'] + cfg['telemetry_extra_features']
    registry = read(base / 'models/registry.json')
    references = pd.read_csv(base / 'results/reference/metrics.csv', dtype={'seed': str})
    comparators = references.groupby(['experiment_id', 'method']).comparator.first().to_dict()
    pieces = []
    for mid, model in registry.items():
        exp, split, window, seed, method = mid.split('/')
        f = features.loc[features.window_id.eq(window) & features.split.eq(split)].copy()
        assert len(f) and split == 'heldout'
        p, b, o, a, terms, intercept = core.predict(model, f, names)
        q = f[['recording_id', 'origin_timestamp', 'target_timestamp', 'cpu_current_c', 'observed_c']].copy()
        q['origin_id'] = q.recording_id + '::' + q.origin_timestamp
        for k, v in dict(experiment_id=exp, run_id=exp, split=split, group_id=f.group,
                         workload_type='inference', seed=int(seed), method=method,
                         comparator=comparators[(exp, method)], model_id=mid, dataset_id='original').items():
            q[k] = v
        q['predicted_temperature_c'] = p
        q['baseline_prediction_c'] = b
        q['original_physics_correction_c'] = o
        q['applied_physics_correction_c'] = a
        q['correction_intercept_c'] = intercept
        for j, name in enumerate(['C', 'R', 'I']):
            q['original_' + name + '_contribution_c'] = terms[:, j]
        q['signed_error_c'] = p - q.observed_c
        q['absolute_error_c'] = abs(q.signed_error_c)
        q['squared_error_c'] = q.signed_error_c ** 2
        lo, hi = model['temperature_range']
        q['inside_training_temperature_range'] = q.cpu_current_c.between(lo, hi)
        q['training_min_c'], q['training_max_c'] = lo, hi
        pieces.append(q)
    prediction = pd.concat(pieces, ignore_index=True)
    out = base / 'local/results'
    out.mkdir(exist_ok=True)
    prediction.to_parquet(out / 'predictions.parquet', index=False)
    metrics = core.metrics(prediction, out)  # Original metric equations and independent check.
    optional = base / 'local/reference_predictions.parquet'
    maximum = None
    if optional.exists():
        old = pd.read_parquet(optional)
        old = old.loc[old.experiment_id.isin(['ridge_original', 'neural_refined']) & old.split.eq('heldout')]
        keys = ['experiment_id', 'split', 'method', 'seed', 'origin_id']
        j = old.merge(prediction, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
        assert len(j) == len(prediction) == len(old)
        maximum = float(abs(j.predicted_temperature_c_old - j.predicted_temperature_c_new).max())
        np.testing.assert_allclose(j.predicted_temperature_c_old, j.predicted_temperature_c_new, atol=1e-10, rtol=0)
    summary = dict(dataset='jetson', raw_sources_verified=20, prepared_rows=len(features),
                   development_rows=2714, heldout_origins=1457, prediction_rows=len(prediction),
                   frozen_aggregate_reference_rows=len(references), aggregate_tolerance_c=1e-10,
                   optional_historical_prediction_max_difference_c=maximum, new_fits=0,
                   source='Original raw recordings and author-selected CSV windows; no prepared inputs used')
    (out / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    head = metrics.loc[metrics.seed.eq('SEED_MEAN') & metrics.scope.eq('aggregate') & metrics.stratum.eq('all')]
    print(head[['experiment_id', 'method', 'mae_c', 'rmse_c']].to_string(index=False))
    return summary


def prepare_intel(source):
    base = ROOT / 'intel'
    local = base / 'local'
    local.mkdir(exist_ok=True)
    manifest = read(base / 'manifests/download_manifest.json')
    for item in manifest:
        # Accept the publisher repository layout or the original manifest-named downloads.
        candidate = source / item['aliases'][0]
        if not candidate.exists():
            candidate = source / Path(item['path']).name
        stage_file(candidate, local / item['path'], item)
    (local / 'sources/download_manifest.json').write_text(json.dumps(manifest))
    (local / 'sources/github_tree.json').write_text(json.dumps({'sha': 'f7f8d62e806ce2b64e93c58fce2bb18604f067ad'}))
    prep = load('intel_preparation', base / 'prepare.py')
    prep.ROOT = local  # Original audit and prepare routines; only output/source root changes.
    prep.audit()
    return prep.prepare()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['jetson', 'intel'])
    parser.add_argument('--source-dir', type=Path, help='Locally obtained publisher repository/root; prepare from original recordings.')
    parser.add_argument('--refit-check', action='store_true', help='Intel only: separate fixed-procedure fitting verification.')
    args = parser.parse_args()
    if args.refit_check and args.dataset != 'intel':
        parser.error('Only the existing Intel fixed-procedure refit check is packaged.')
    if args.source_dir:
        source = args.source_dir.resolve()
        features = prepare_jetson(source) if args.dataset == 'jetson' else prepare_intel(source)
    else:
        import pandas as pd
        path = ROOT / args.dataset / 'local/features.parquet'
        if not path.exists():
            parser.error('No local features. Supply --source-dir; see README.md for publisher downloads.')
        features = pd.read_parquet(path)
    if args.dataset == 'jetson':
        evaluate_jetson(features)
    else:
        import subprocess
        command = [sys.executable, '-B', str(ROOT / 'intel/run.py')]
        if args.refit_check:
            # Replay first, then independently check fixed-procedure fits.
            subprocess.run(command, check=True)
            command += ['--refit-check']
        subprocess.run(command, check=True)
    print('Results: ' + str(ROOT / args.dataset / 'local/results'))


if __name__ == '__main__':
    main()
