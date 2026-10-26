"""Canonical, unmodified-rate telemetry and verified author-selected windows.

Offline only. No downloaded code execution, interpolation or model fitting.
"""
import csv
import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from acquire import verify
from audit import ROOT, STAMP, raw_fields, number

TEMPERATURES = ('cpu', 'soc0', 'soc1', 'soc2', 'tj', 'tboard', 'tdiode', 'gpu')
RAILS = ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0', 'VDDQ_VDD2_1V8AO')
DERIVED = ROOT / 'derived'
REPORT = ROOT / 'reports/stage2'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def schema_definition():
    fields = []
    def add(name, kind, unit, meaning):
        fields.append(dict(name=name, type=kind, unit=unit, nullable=True, meaning=meaning))
    for name, meaning in {
        'source_path': 'Unchanged raw source path relative to repository',
        'source_sha256': 'SHA-256 of entire original source file',
        'recording_id': 'Capture artifact identity, not independent experiment identity',
        'timestamp_text': 'Original raw wall-clock text; timezone unknown',
        'timestamp_timezone': 'Always unknown; preamble UTC label does not resolve sample timezone',
        'author_window_id': 'CSV-derived window membership only; null on all unselected raw tails',
        'filename_workload_hint': 'Filename association only, NOT confirmed per-sample workload',
        'filename_mode_hint': 'Filename association only, NOT measured or confirmed power mode',
        'confirmed_workload': 'Unknown for every sample; no execution markers',
        'confirmed_mode': 'Unknown for every sample; no nvpmodel snapshot',
    }.items():
        add(name, 'string', None, meaning)
    add('raw_sample_index', 'int64', 'sample', 'Zero-based telemetry sample index, excluding preamble')
    add('source_line_number', 'int64', 'line', 'One-based original text line number')
    add('wall_time', 'timestamp[us]', None, 'Timezone-naive parsed wall clock; never localized as UTC')
    add('elapsed_s', 'float64', 's', 'Wall-clock difference from first raw sample')
    add('interval_s', 'float64', 's', 'Difference from preceding raw sample; first sample null')
    for s in TEMPERATURES:
        add('temperature_'+s+'_c', 'float64', 'degC', 'Named '+s+' sensor; tboard is not ambient')
    for rail in RAILS:
        for kind in ('instantaneous', 'average'):
            add('power_'+rail+'_'+kind+'_mw', 'float64', 'mW',
                'Separate rail reading; no rail summation; average horizon unknown' if kind=='average'
                else 'Present sampled power, not electrical current; rail scope unverified')
    for i in range(12):
        add(f'cpu{i}_active', 'bool', None, 'True=active, false=explicit off, null=unobserved')
        add(f'cpu{i}_utilization_pct', 'float64', '%', 'Null for off/unobserved; never replaced with zero')
        add(f'cpu{i}_clock_mhz', 'float64', 'MHz', 'Null for off/unobserved')
    add('gpu_utilization_pct', 'float64', '%', 'GR3D load; not GPU clock')
    add('gpu_clock_mhz', 'float64', 'MHz', 'Only if explicitly observed; missing in these sources')
    add('ram_used_mb', 'float64', 'MB (source label)', 'Reported RAM usage')
    add('ram_total_mb', 'float64', 'MB (source label)', 'Reported RAM total; not proof of device SKU')
    return dict(version=1, fields=fields, transformations='Parsing and unit-explicit renaming only; no resampling, interpolation, smoothing or imputation.',
                missing='Parquet null, empty CSV cell; unavailable ambient/fan/throttle are not invented columns.',
                configuration='Hints only; documented associations and confidence live in windows.json; physical device identity unresolved.',
                time='Naive wall-clock comparisons assume a shared local clock for grouping, not verified session identity.',
                source_manifest='manifests/sources.json')


def arrow_schema():
    types = {'string': pa.string(), 'float64': pa.float64(), 'int64': pa.int64(),
             'bool': pa.bool_(), 'timestamp[us]': pa.timestamp('us')}
    return pa.schema([pa.field(f['name'], types[f['type']], metadata={
        b'unit': (f['unit'] or '').encode(), b'meaning': f['meaning'].encode()})
        for f in schema_definition()['fields']], metadata={b'timestamp_timezone': b'unknown', b'stage': b'2'})


def author_rows(record):
    with (ROOT / record['csv_path']).open(newline='') as f:
        reader = csv.reader(f)
        next(reader)
        header = next(reader)
        rows = []
        for position, row in enumerate(reader, 1):
            if not row:
                if record['recording_id'] != 'resnet_maxn' or position != 1:
                    raise ValueError('Unexpected blank CSV record')
                continue
            if len(row) != len(header):
                raise ValueError('Unexpected malformed CSV record')
            rows.append(dict(zip(header, map(number, row))))
    indices = [int(r['Index']) for r in rows]
    if indices != list(range(indices[0], indices[-1]+1)):
        raise ValueError('Noncontiguous author window: requires explicit review')
    return {int(r['Index']): r for r in rows}


def parse_sample(line):
    match = STAMP.match(line)
    if not match:
        raise ValueError('Unexpected non-telemetry line after preamble')
    stamp = match[1]
    fields = raw_fields(line)
    out = dict(timestamp_text=stamp, wall_time=datetime.strptime(stamp, '%m-%d-%Y %H:%M:%S'),
               timestamp_timezone='unknown')
    for s in TEMPERATURES:
        out['temperature_'+s+'_c'] = fields.get(s+' Temperature (C)')
    for rail in RAILS:
        for dest, old in [('instantaneous', 'Current'), ('average', 'Average')]:
            out[f'power_{rail}_{dest}_mw'] = fields.get(f'{rail} {old} (mW)')
    for i in range(12):
        load = fields.get(f'CPU {i} Load (%)')
        if f'CPU {i} malformed' in fields:
            raise ValueError('Malformed CPU token')
        out[f'cpu{i}_active'] = None if load is None else load != 'off'
        out[f'cpu{i}_utilization_pct'] = None if load == 'off' else load
        out[f'cpu{i}_clock_mhz'] = fields.get(f'CPU {i} Frequency (MHz)')
    out.update(gpu_utilization_pct=fields.get('Used GR3D (%)'),
               gpu_clock_mhz=fields.get('GR3D Frequency (MHz)'),
               ram_used_mb=fields.get('Used RAM (MB)'),ram_total_mb=fields.get('Total RAM (MB)'))
    return out, fields


def verify_sources():
    entries = json.loads((ROOT/'manifests/sources.json').read_text())
    unique = {e['local_path']: e for e in entries}
    for path, e in unique.items():
        if verify((ROOT/path).read_bytes(),e) != e['sha256']:
            raise ValueError('Source SHA mismatch: '+path)
    return unique


def make_window(record, selected, csv_rows):
    first, last = selected[0], selected[-1]
    rid = record['recording_id']
    status = 'candidate'
    reason = 'Retain complete author-selected window; no confirmed execution boundary or independent trial.'
    if record['category'] != 'inference':
        status, reason = 'outside_primary', 'Training or unclassified resnet_test; not part of inference study.'
    if rid == 'depth_maxn':
        status, reason = 'quarantine', '882 s aligned span versus 190.54 s documented runtime; selected overlap with Depth 50W.'
    if rid == 'depth_50W':
        status, reason = 'quarantine', 'Complete selected window overlaps Depth MAXN; concurrent logging/activity unresolved. Own duration is plausible, but cleanliness is not established.'
    return dict(window_id=rid+'__author', recording_id=rid, window_type='author-selected',
                source_path=record['raw_path'],source_sha256=first['source_sha256'],
                csv_source_path=record['csv_path'],raw_start_index=first['raw_sample_index'],raw_end_index=last['raw_sample_index'],
                raw_start_timestamp=first['timestamp_text'],raw_end_timestamp=last['timestamp_text'],
                wall_start=first['wall_time'].isoformat(),wall_end=last['wall_time'].isoformat(),timezone='unknown',
                sample_count=len(selected),elapsed_duration_s=(last['wall_time']-first['wall_time']).total_seconds(),
                raw_capture_start=record['raw_first_timestamp'],raw_capture_end=record['raw_last_timestamp'],
                raw_sample_count=record['raw_samples'],category=record['category'],
                documented_workload=record['workload_documented'],nominal_mode=record['nominal_mode_filename'],
                metadata_confidence='Document section plus filename association; not sample-level execution/configuration confirmation',
                metadata_source='sources/scales-docs/docs/ML_Training.md' if record['category']=='training' else 'sources/scales-docs/docs/ML_Jetson_Performance.md',
                device_information=record['device_documented'],inclusion_status=status,inclusion_rationale=reason,
                timing_flags=dict(gaps_over_1s=sum(x['interval_s']>1 for x in selected[1:]),
                                  max_internal_interval_s=max(x['interval_s'] for x in selected[1:]),
                                  csv_uses_uniform_index_time=True,timezone_unknown=True),
                data_quality_flags=['no_workload_markers','no_gpu_clock','no_ambient_fan_throttle','uncertain_device_mode']+
                                   (['blank_csv_record_excluded'] if record['malformed_csv_rows'] else []),
                selected_window_overlaps=[],full_raw_period_overlaps=[],same_date_windows=[],
                dependence_flags=['date_group_is_assumption','possible_thermal_carryover','physical_device_identity_unknown'],
                proposed_validation_group='wall_date_'+first['wall_time'].date().isoformat(),
                index_mapping='Every author CSV row verified at its original raw index on Stage 1 signature columns and anonymous soc1 temperature; no clipping.',
                alternative_window=False)


def prepare_record(record, source):
    rid = record['recording_id']
    selected_csv = author_rows(record)
    schema = arrow_schema()
    path = DERIVED/'canonical'/f'{rid}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.parquet.part')
    rows, selected, preview = [], [], []
    first = previous = None
    index = 0
    with pq.ParquetWriter(temp, schema, compression='zstd') as writer:
        with (ROOT/record['raw_path']).open() as f:
            next(f)  # Preserved preamble in Stage 1 manifest, not a telemetry row.
            for line_number, line in enumerate(f, 2):
                row, original_fields = parse_sample(line)
                stamp = row['wall_time']
                if first is None:
                    first = stamp
                interval = None if previous is None else (stamp-previous).total_seconds()
                if interval is not None and interval <= 0:
                    raise ValueError('Non-increasing source time; requires review')
                row.update(source_path=record['raw_path'],source_sha256=source['sha256'],recording_id=rid,
                           raw_sample_index=index,source_line_number=line_number,elapsed_s=(stamp-first).total_seconds(),
                           interval_s=interval,author_window_id=rid+'__author' if index in selected_csv else None,
                           filename_workload_hint=record['workload_filename'],filename_mode_hint=record['nominal_mode_filename'],
                           confirmed_workload=None,confirmed_mode=None)
                if index in selected_csv:
                    csv_row = selected_csv[index]
                    for key in record['alignment']['signature_columns']:
                        if csv_row[key] != original_fields.get(key):
                            raise ValueError(f'Author alignment mismatch: {rid}:{index}:{key}')
                    if csv_row[' Temperature (C)'] != row['temperature_soc1_c']:
                        raise ValueError('Anonymous temperature does not match soc1')
                    selected.append(row.copy())
                if len(preview)<5:
                    preview.append(row.copy())
                rows.append(row)
                if len(rows)==10000:
                    writer.write_table(pa.Table.from_pylist(rows,schema=schema))
                    rows=[]
                previous=stamp
                index+=1
        if rows:
            writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    if index != record['raw_samples'] or len(selected)!=len(selected_csv):
        raise ValueError('Raw or selected sample count mismatch')
    temp.replace(path)
    preview_path=DERIVED/'preview'/f'{rid}.csv'
    preview_path.parent.mkdir(exist_ok=True)
    with preview_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=schema.names);w.writeheader();w.writerows(preview)
    window=make_window(record,selected,selected_csv)
    return window, dict(recording_id=rid,path=str(path.relative_to(ROOT)),rows=index,
                        bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_path=record['raw_path'],source_sha256=source['sha256'])


def annotate_dependence(windows):
    for a in windows:
        for b in windows:
            if a is b:
                continue
            if max(a['wall_start'],b['wall_start']) <= min(a['wall_end'],b['wall_end']):
                a['selected_window_overlaps'].append(b['window_id'])
            if max(a['raw_capture_start'],b['raw_capture_start']) <= min(a['raw_capture_end'],b['raw_capture_end']):
                a['full_raw_period_overlaps'].append(b['recording_id'])
            if a['proposed_validation_group']==b['proposed_validation_group']:
                a['same_date_windows'].append(b['window_id'])
        if a['selected_window_overlaps']:
            a['dependence_flags'].append('selected_window_overlap')
        if a['full_raw_period_overlaps']:
            a['dependence_flags'].append('full_capture_overlap_not_equivalent_to_selected_overlap')


def main():
    sources=verify_sources()
    print(f'Integrity verified: {len(sources)} existing files; no acquisition.',flush=True)
    records=json.loads((ROOT/'manifests/recordings.json').read_text())
    windows, canonical = [], []
    for record in records:
        window, data=prepare_record(record,sources[record['raw_path']])
        windows.append(window);canonical.append(data)
        print(record['recording_id'],data['rows'],window['inclusion_status'],flush=True)
    annotate_dependence(windows)
    dump(ROOT/'manifests/windows.json',windows)
    dump(ROOT/'manifests/canonical-schema.json',schema_definition())
    dump(ROOT/'manifests/canonical-files.json',canonical)
    import importlib.metadata
    dump(REPORT/'environment.json',dict(python=sys.version,platform=platform.platform(),
         dependencies={name:importlib.metadata.version(name) for name in ('pyarrow','numpy','pandas','matplotlib')},
         source_manifest_sha256=hashlib.sha256((ROOT/'manifests/sources.json').read_bytes()).hexdigest(),
         command='python scripts/prepare_stage2.py',no_acquisition=True))


if __name__=='__main__':
    main()
