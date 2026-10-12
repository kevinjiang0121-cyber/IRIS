#!/usr/bin/env python3
"""Prepare a local HF review bundle from ONLY the pinned public ZIP; never upload."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

ZIP_SHA = '4ccf4a4b833ad456d8ed80157458f688bd54be98ba810e5bcb2cc59d357fed19'
BASE = 'data/IRIS_data_anonymous/'
COUNTS = {'headline_metrics': 60, 'operative_UR_intervals': 60, 't1_t3': 6,
          'iaa': 12, 'paired_reading': 12, 'backtranslation_control': 12}
TEXT = {'model', 'condition', 'metric', 'language'}
INT = {'n', 'operative_count', 'harm_outside_operative', 'n00', 'n01', 'n10', 'n11'}
BEHAVIOR_FIELDS = ['BehaviorID', 'FunctionalCategory', 'SemanticCategory', 'Tags', 'Source',
                   'Behavior_EN', 'ContextString_EN', 'Behavior_EO', 'ContextString_EO',
                   'Behavior_TK', 'ContextString_TK']

def require(ok, message):
    if not ok:
        raise ValueError(message)

def convert(archive, out):
    blob = archive.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == ZIP_SHA, 'Unreviewed ZIP: audit new revision first')
    require(not out.exists(), 'Output already exists; choose a new directory')
    tables = {}
    provenance = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        manifest = json.loads(z.read(BASE + 'MANIFEST_SHA256.json'))
        require(set(z.namelist()) == {BASE + p for p in manifest} | {BASE + 'MANIFEST_SHA256.json'},
                'Unexpected archive members')
        for path, meta in manifest.items():
            b = z.read(BASE + path)
            require(len(b) == meta['bytes'] and hashlib.sha256(b).hexdigest() == meta['sha256'],
                    'Archive manifest mismatch: ' + path)
        def read_csv(path):
            reader = csv.DictReader(io.StringIO(z.read(BASE + path).decode('utf-8-sig')))
            rows = list(reader)
            require(len(reader.fieldnames) == len(set(reader.fieldnames)), 'Duplicate headers')
            require(all(None not in r and all(v is not None for v in r.values()) for r in rows), 'Ragged CSV')
            provenance[path] = manifest[path]
            return reader.fieldnames, rows
        fields, behaviors = read_csv('data/behavior_datasets/behavior_all.csv')
        require(fields == BEHAVIOR_FIELDS, 'Behavior schema changed')
        ids = [r['BehaviorID'] for r in behaviors]
        require(len(ids) == len(set(ids)) == 105, 'Behavior ID mismatch')
        require(set(ids) == set(z.read(BASE + 'data/splits/behavior_ids_full105.txt').decode().splitlines()), 'Split mismatch')
        sources = dict(collections.Counter(r['Source'] for r in behaviors))
        require(sources == {'HarmBench': 33, 'JailbreakBench': 52, 'StrongReject/rewrite': 20}, 'Sources changed')
        tables['behaviors'] = behaviors
        assets = []
        for variant, path in [('FormalLogic', 'data/full105/formal_logic_translations_merged.json'),
                              ('Yoruba', 'data/full105/low_resource_translations_yoruba.json'),
                              ('Zulu', 'data/full105/low_resource_translations_zulu.json')]:
            values = json.loads(z.read(BASE + path))
            require(set(values) == set(ids), 'Asset coverage mismatch')
            require(all(isinstance(v, str) and v for v in values.values()), 'Invalid asset text')
            provenance[path] = manifest[path]
            for bid in ids:
                assets.append({'BehaviorID': bid, 'variant': variant, 'text': values[bid],
                               'source_file': path, 'license_status': 'pending_component_review'})
        tables['supplementary_inputs'] = assets
        for name, count in COUNTS.items():
            _, rows = read_csv('results/current_submission/' + name + '.csv')
            require(len(rows) == count, 'Unexpected result count: ' + name)
            keys = []
            for row in rows:
                for k, v in row.items():
                    if k not in TEXT:
                        row[k] = int(v) if k in INT else float(v)
                        require(math.isfinite(row[k]), 'Nonfinite result')
                keys.append(tuple(row[k] for k in ['model', 'condition', 'metric', 'language'] if k in row))
                if 'n' in row:
                    require(row['n'] == 105, 'Unexpected denominator')
                if all(k in row for k in ['n00', 'n01', 'n10', 'n11']):
                    require(sum(row[k] for k in ['n00', 'n01', 'n10', 'n11']) == row['n'], 'Contingency mismatch')
            require(len(keys) == len(set(keys)), 'Duplicate result keys')
            tables[name] = rows
        headline = {(r['model'], r['condition']): r for r in tables['headline_metrics']}
        require(len({r['model'] for r in headline.values()}) == 6, 'Model count mismatch')
        require(len({r['condition'] for r in headline.values()}) == 10, 'Condition count mismatch')
        for r in tables['operative_UR_intervals']:
            h = headline[(r['model'], r['condition'])]
            require(abs(r['operative_UR'] - h['operative_UR']) < 1e-8, 'Point estimates disagree')
            require(0 <= r['ci_low'] <= r['operative_UR'] <= r['ci_high'] <= 100, 'Invalid interval')
        for h in headline.values():
            require(abs(h['operative_UR'] - 100 * h['operative_count'] / h['n']) < 1e-8, 'Count/rate mismatch')
            for k in ['operative_UR', 'recognition_rate', 'ASR', 'Shift', 'B']:
                require(0 <= h[k] <= 100, 'Out-of-range headline rate')
    # Heuristic triage, never a claim that no identifying information exists.
    findings = []
    patterns = {'email': r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
                'token': r'(?:sk-|hf_)[A-Za-z0-9]{20,}',
                'phone': r'(?<!\w)(?:\+\d[\d ()-]{8,}\d)(?!\w)'}
    for name, rows in tables.items():
        for i, row in enumerate(rows):
            for field, value in row.items():
                if isinstance(value, str):
                    for kind, pattern in patterns.items():
                        if re.search(pattern, value):
                            findings.append({'table': name, 'row': i, 'field': field, 'kind': kind})
    require(not findings, 'Potential sensitive values found; review before conversion')
    out.mkdir(parents=True)
    (out / 'data').mkdir()
    schemas = {}
    for name, rows in tables.items():
        target = out / 'data' / (name + '.jsonl')
        target.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
        schemas[name] = {k: type(v).__name__ for k, v in rows[0].items()}
    card = Path(__file__).resolve().parents[1] / 'docs/huggingface/DATASET_CARD.md'
    (out / 'README.md').write_bytes(card.read_bytes())
    for name in ['LICENSE_DATA.md', 'STRONGREJECT_PROVENANCE.csv']:
        (out / name).write_bytes(card.with_name(name).read_bytes())
    report = {'archive_sha256': ZIP_SHA, 'release_status': 'DRAFT_NOT_AUTHORIZED_FOR_UPLOAD',
              'source_counts': sources, 'row_counts': {n: len(r) for n, r in tables.items()},
              'schemas': schemas, 'source_files': provenance, 'sensitive_scan_findings': findings,
              'manual_review_required': ['Names embedded in BehaviorID (e.g. maria, Evan)',
                  'All text fields including multilingual text and contextual identifiers',
                  'Per-item upstream provenance and IRIS translation/aggregate rights'],
              'outputs': {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(out.rglob('*')) if p.is_file()}}
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['row_counts'], sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('artifacts/data.zip'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.archive, args.output)
