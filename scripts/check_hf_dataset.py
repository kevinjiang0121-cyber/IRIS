#!/usr/bin/env python3
"""Validate JSONL loading, declared configs, hashes and exact source round trips."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import yaml
from datasets import load_dataset
from prepare_hf_dataset import BASE, COUNTS, ZIP_SHA, require, export_text

bundle = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
report = json.loads((bundle / 'conversion_report.json').read_text())
require(report['archive_sha256'] == ZIP_SHA, 'Wrong source revision')
for name, sha in report['outputs'].items():
    require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == sha, 'Output mutated: ' + name)
meta = yaml.safe_load((bundle / 'README.md').read_text().split('---', 2)[1])
require(meta.get('license') == 'other' and meta.get('license_link') == 'https://huggingface.co/datasets/ArthurTThomas/IRIS/blob/main/LICENSE_DATA.md',
        'Mixed component licenses must be stated accurately')
expected_files = {'README.md', 'LICENSE_DATA.md', 'SOURCE_PROVENANCE.csv', 'conversion_report.json'} | {
    spec['path'] for c in meta['configs'] for spec in c['data_files']}
actual_files = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
require(actual_files == expected_files, 'Unexpected or missing upload files')
require(set(report['outputs']) == expected_files - {'conversion_report.json'}, 'Incomplete output hash index')
require(not any(p.is_symlink() for p in bundle.rglob('*')), 'Symlinks are not upload files')
with zipfile.ZipFile(repo / 'artifacts/data.zip') as z:
    require({c['config_name'] for c in meta['configs']} == set(COUNTS) | {'behaviors', 'supplementary_inputs'},
            'Missing or unexpected configuration')
    for config in meta['configs']:
        name = config['config_name']
        spec = config['data_files'][0]
        rows = list(load_dataset(str(bundle), name=name, split=spec['split']))
        require(len(rows) == report['row_counts'][name], 'Loaded row count mismatch')
        if name == 'behaviors':
            source = list(csv.DictReader(io.StringIO(z.read(BASE + 'data/behavior_datasets/behavior_all.csv').decode('utf-8-sig'))))
            expected = [{k: (v if k == 'BehaviorID' else export_text(r['BehaviorID'], v))
                         for k, v in r.items()} for r in source]
            require(rows == expected, 'Unexpected behavior text change')
        elif name in COUNTS:
            source = list(csv.DictReader(io.StringIO(z.read(BASE + 'results/current_submission/' + name + '.csv').decode('utf-8-sig'))))
            require(len(rows) == len(source), 'Source result row count mismatch')
            for row, original in zip(rows, source):
                require(set(row) == set(original), 'Result schema mismatch')
                for k, v in original.items():
                    require(row[k] == (v if isinstance(row[k], str) else float(v)), 'Metric value changed')
        else:
            for row in rows:
                source = json.loads(z.read(BASE + row['source_file']))
                require(row['text'] == export_text(row['BehaviorID'], source[row['BehaviorID']]), 'Unexpected supplementary text change')
        print(name, len(rows), 'OK')
provenance = list(csv.DictReader((bundle / 'SOURCE_PROVENANCE.csv').open()))
require(len(provenance) == len({r['BehaviorID'] for r in provenance}) == 105, 'Incomplete source provenance')
require({r['BehaviorID'] for r in provenance} == {r['BehaviorID'] for r in load_dataset(str(bundle), name='behaviors', split='evaluation')}, 'Source provenance IDs disagree')
require(report['release_status'] == 'PREPARED_NOT_UPLOADED', 'Draft status changed')
print('All eight configurations load; source values match the documented export transformation. No upload performed.')
