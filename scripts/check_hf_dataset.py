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
from prepare_hf_dataset import BASE, COUNTS, ZIP_SHA, require

bundle = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
report = json.loads((bundle / 'conversion_report.json').read_text())
require(report['archive_sha256'] == ZIP_SHA, 'Wrong source revision')
for name, sha in report['outputs'].items():
    require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == sha, 'Output mutated: ' + name)
meta = yaml.safe_load((bundle / 'README.md').read_text().split('---', 2)[1])
require('license' not in meta, 'Do not add an unreviewed blanket license')
with zipfile.ZipFile(repo / 'artifacts/data.zip') as z:
    for config in meta['configs']:
        name = config['config_name']
        spec = config['data_files'][0]
        rows = list(load_dataset('json', data_files={spec['split']: str(bundle / spec['path'])}, split=spec['split']))
        require(len(rows) == report['row_counts'][name], 'Loaded row count mismatch')
        if name == 'behaviors':
            source = list(csv.DictReader(io.StringIO(z.read(BASE + 'data/behavior_datasets/behavior_all.csv').decode('utf-8-sig'))))
            require(rows == source, 'Behavior text changed')
        elif name in COUNTS:
            source = list(csv.DictReader(io.StringIO(z.read(BASE + 'results/current_submission/' + name + '.csv').decode('utf-8-sig'))))
            for row, original in zip(rows, source):
                for k, v in original.items():
                    require(row[k] == (v if isinstance(row[k], str) else float(v)), 'Metric value changed')
        else:
            for row in rows:
                source = json.loads(z.read(BASE + row['source_file']))
                require(row['text'] == source[row['BehaviorID']], 'Supplementary text changed')
        print(name, len(rows), 'OK')
require(report['release_status'] == 'DRAFT_NOT_AUTHORIZED_FOR_UPLOAD', 'Draft status changed')
print('All eight configurations load; exact source-value round trips pass. Upload remains on hold.')
