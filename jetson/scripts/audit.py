"""Read-only telemetry audit. No resampling, imputation, fitting or modelling."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = re.compile(r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) ')
TEMPS = re.compile(r'\b(\w+)@(-?[\d.]+)C')
POWER = re.compile(r'\b(\w+) (\d+)mW/(\d+)mW')


def raw_fields(line):
    d = {}
    for name, val in TEMPS.findall(line):
        d[name + ' Temperature (C)'] = float(val)
    for name, current, avg in POWER.findall(line):
        d[name + ' Current (mW)'] = float(current)
        d[name + ' Average (mW)'] = float(avg)
    m = re.search(r'RAM (\d+)/(\d+)MB', line)
    if m:
        d['Used RAM (MB)'], d['Total RAM (MB)'] = map(float, m.groups())
    m = re.search(r'GR3D_FREQ (\d+)%', line)
    if m:
        d['Used GR3D (%)'] = float(m[1])
    m = re.search(r'GR3D_FREQ \d+%@(?:\[)?(\d+)', line)
    if m:
        d['GR3D Frequency (MHz)'] = float(m[1])
    m = re.search(r'CPU \[([^]]+)\]', line)
    if m:
        for i, token in enumerate(m[1].split(',')):
            if token == 'off':
                d[f'CPU {i} Load (%)'] = 'off'
            else:
                pair = re.fullmatch(r'(\d+)%@(\d+)', token)
                if pair:
                    d[f'CPU {i} Load (%)'] = float(pair[1])
                    d[f'CPU {i} Frequency (MHz)'] = float(pair[2])
                else:
                    d[f'CPU {i} malformed'] = token
    return d


def number(x):
    if x in ('', 'off'):
        return x
    try:
        return float(x)
    except ValueError:
        return x


def timing(values):
    delta = [b-a for a,b in zip(values, values[1:])]
    return dict(first=values[0] if values else None, last=values[-1] if values else None,
                elapsed_seconds=values[-1]-values[0] if values else None,
                delta_seconds=dict(Counter(delta)), duplicates=sum(x==0 for x in delta),
                resets=sum(x<0 for x in delta), gaps_over_one_second=sum(x>1 for x in delta))


def audit_pair(path):
    with path.open(newline='') as f:
        reader = csv.reader(f)
        preamble = next(reader)
        headers = next(reader)
        rows = list(reader)
    malformed = [i+1 for i,r in enumerate(rows) if len(r)!=len(headers)]
    good = [dict(zip(headers, map(number,r))) for r in rows if len(r)==len(headers)]
    columns = {}
    for h in headers:
        vals = [r[h] for r in good]
        numeric = [v for v in vals if isinstance(v,float)]
        columns[h] = dict(missing=vals.count(''), off=vals.count('off'),
                          malformed=sum(not isinstance(v,float) and v not in ('','off') for v in vals),
                          distinct=len(set(vals)), min=min(numeric) if numeric else None,
                          max=max(numeric) if numeric else None)
    raw = path.with_suffix('.txt')
    if not raw.exists():
        raw = path.with_suffix('.txt--graph')
    # Exact shared telemetry signature: RAM, GPU load, named temperatures and every power pair.
    keys = [h for h in headers if h in ('Used RAM (MB)','Total RAM (MB)','Used GR3D (%)')
            or (h.endswith('Temperature (C)') and not h.startswith(' ')) or h.endswith('(mW)')
            or re.fullmatch(r'CPU \d+ Load \(%\)', h)]
    wanted = defaultdict(list)
    for i,r in enumerate(good):
        wanted[tuple(r[k] for k in keys)].append(i)
    matches = [[] for _ in good]
    anonymous = [set(('soc0','soc1','soc2')) for _ in good]
    stats = defaultdict(Counter)
    times = []
    nontelemetry = []
    malformed_raw = []
    raw_header = None
    nontelemetry_count = 0
    optional_markers = Counter()
    index_matches = []
    index_anonymous = set(('soc0','soc1','soc2'))
    by_index = {int(r['Index']): r for r in good}
    n = 0
    with raw.open() as f:
        for line_no,line in enumerate(f,1):
            for token in ('fan', 'ambient', 'throttl', 'workload', 'epoch'):
                if token in line.lower():
                    optional_markers[token] += 1
            m = STAMP.match(line)
            if not m:
                nontelemetry_count += 1
                if raw_header is None:
                    raw_header = line.strip()
                if len(nontelemetry)<20:
                    nontelemetry.append(dict(line=line_no,text=line.strip()[:300]))
                continue
            try:
                dt = datetime.strptime(m[1], '%m-%d-%Y %H:%M:%S')
            except ValueError:
                malformed_raw.append(line_no)
                continue
            times.append((dt-datetime(1970,1,1)).total_seconds())
            fields = raw_fields(line)
            candidate = by_index.get(n)
            if candidate and all(candidate[k] == fields.get(k) for k in keys):
                index_matches.append(n)
                index_anonymous &= {s for s in ('soc0','soc1','soc2') if fields.get(s+' Temperature (C)') == candidate.get(' Temperature (C)')}
            for k,v in fields.items():
                stats[k][v] += 1
            if not all(k in fields for k in keys):
                malformed_raw.append(line_no)
            for i in wanted.get(tuple(fields.get(k) for k in keys), []):
                matches[i].append(n)
                anonymous[i] &= {s for s in ('soc0','soc1','soc2')
                                 if fields.get(s+' Temperature (C)') == good[i].get(' Temperature (C)')}
            n += 1
    chosen = [m[0] for m in matches if len(m)==1]
    aligned_times = [times[m[0]] for m in matches if len(m)==1]
    raw_summary = {}
    for k,c in stats.items():
        nums = [v for v in c if isinstance(v,float)]
        raw_summary[k] = dict(present=sum(c.values()),missing=n-sum(c.values()),off=c.get('off',0),
                              distinct=len(c),min=min(nums) if nums else None,max=max(nums) if nums else None)
    stem=path.stem
    kind = 'training' if 'training' in stem else ('unclassified' if stem=='resnet_test' else 'inference')
    mode = stem.split('_')[-1] if '_' in stem else None
    if mode in ('training','test'):
        mode=None
    return dict(recording_id=stem,category=kind,workload_filename=stem.split('_')[0],
                nominal_mode_filename=mode,metadata_confidence='filename; see documentation mapping in DATA_AUDIT.md',
                device_documented='Jetson AGX Orin; conflicting 32GB/64GB descriptions; serial and exact configuration unavailable',
                workload_documented=('ResNet-50' if kind=='training' else {'resnet':'ResNet-152','depth':'Depth Anything V2','phi':'Phi-3.5'}.get(stem.split('_')[0])) if kind!='unclassified' else None,
                repeat_trial_confirmed=False,independence_status='distinct file pair, not proof of independent experimental trial',
                csv_path=str(path.relative_to(ROOT)),raw_path=str(raw.relative_to(ROOT)),
                csv_preamble=preamble,raw_preamble=raw_header,
                csv_rows=len(rows),structurally_valid_rows=len(good),malformed_csv_rows=malformed,
                usable_temperature_rows=sum(isinstance(r.get('cpu Temperature (C)'),float) for r in good),
                csv_timing=timing([r['Time (mS)']/1000 for r in good]),csv_columns=columns,
                raw_samples=n,raw_timing=timing(times),raw_timestamp_format='%m-%d-%Y %H:%M:%S; timezone unspecified',
                raw_nontelemetry_examples=nontelemetry,raw_nontelemetry_count=nontelemetry_count,
                optional_marker_line_counts=dict(optional_markers),
                raw_first_timestamp=datetime.fromtimestamp(times[0], timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if times else None,
                raw_last_timestamp=datetime.fromtimestamp(times[-1], timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if times else None,
                malformed_raw_lines=malformed_raw,raw_channels=raw_summary,
                alignment=dict(signature_columns=keys,matched_rows=sum(bool(m) for m in matches),
                               index_verified_rows=len(index_matches),index_aligned_timing=timing([times[i] for i in index_matches]),
                               index_aligned_anonymous_sensor_candidates=sorted(index_anonymous),
                               unique_matches=sum(len(m)==1 for m in matches),ambiguous_matches=sum(len(m)>1 for m in matches),
                               unmatched_csv_rows=[i for i,m in enumerate(matches) if not m],
                               first_raw_index=min(chosen) if chosen else None,last_raw_index=max(chosen) if chosen else None,
                               contiguous_unique=len(chosen)==len(good) and all(b-a==1 for a,b in zip(chosen,chosen[1:])),
                               csv_index_offsets= dict(Counter(good[i]['Index']-m[0] for i,m in enumerate(matches) if len(m)==1)),
                               matched_raw_timing=timing(aligned_times),
                               anonymous_sensor_candidates=sorted(set.intersection(*[anonymous[i] for i,m in enumerate(matches) if m])) if any(matches) else []),
                unavailable_or_unestablished=['ambient temperature','fan setting/RPM','explicit throttle flags',
                                             'explicit workload start/end markers','device serial','power-mode configuration file'])


def main():
    results=[]
    for p in sorted((ROOT/'sources/scales-docs/docs/data').glob('*.csv')):
        result=audit_pair(p)
        results.append(result)
        print(p.stem,result['csv_rows'],result['raw_samples'],result['alignment']['matched_rows'],flush=True)
    (ROOT/'manifests/recordings.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['| Recording | Class | CSV rows | CSV span s | Raw samples | Raw span s | Raw gaps >1s | Matched CSV rows |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['recording_id']} | {r['category']} | {r['csv_rows']} | {r['csv_timing']['elapsed_seconds']} | {r['raw_samples']} | {r['raw_timing']['elapsed_seconds']} | {r['raw_timing']['gaps_over_one_second']} | {r['alignment']['matched_rows']} |")
    (ROOT/'reports/recordings.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
