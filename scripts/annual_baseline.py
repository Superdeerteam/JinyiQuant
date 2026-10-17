"""Causal 2022 fixed-weight and monthly-refit diagnosis; current recipe unchanged."""
import argparse
import copy
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.config import Period, RunConfig
from jinyiquant.data import build_batch, load_data, select_targets
from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, write_json
from jinyiquant.runner import _answer_coverage_check, inference_keys
from jinyiquant.tracking import Recorder, capture_source, environment, finish, resource_usage
from scripts.run_seed_ensemble import fit_worker, record_data

ROOT = Path(__file__).resolve().parents[1]
EXP = 'annual-baseline-2022-20261017'


def validate_window(window):
    if not (pd.Timestamp(window['train_end_exclusive']) <= pd.Timestamp(window['evaluation_start'])
            and pd.Timestamp(window['evaluation_start']).year == 2022
            and pd.Timestamp(window['evaluation_end_exclusive']) <= pd.Timestamp('2023-01-01')
            and len(window['train_dates']) == 10):
        raise ValueError('Noncausal weights, wrong training length or reserved evaluation period')


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    base = read_json(ROOT / 'configs/fixed_seed_ensemble_20261017_plan.json')['configs']['research_a']
    calendar = pd.read_parquet(base['data']['instruments_path'], columns=['date'])
    dates = sorted(pd.to_datetime(calendar.date).dt.normalize().unique())
    dates = pd.DatetimeIndex(dates)
    windows, configs = [], {}
    for month in range(1, 13):
        start = pd.Timestamp(2022, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        training = dates[dates < start][-10:]
        evaluation = dates[(dates >= start) & (dates < end)]
        if len(training) != 10 or len(evaluation) < 10:
            raise ValueError('Incomplete calendar')
        name = start.strftime('%Y%m')
        w = dict(name=name, train_start=str(training[0].date()),
                 train_end_exclusive=str(start.date()), train_dates=training.strftime('%Y-%m-%d').tolist(),
                 evaluation_start=str(start.date()), evaluation_end_exclusive=str(end.date()),
                 evaluation_dates=evaluation.strftime('%Y-%m-%d').tolist())
        validate_window(w)
        raw = copy.deepcopy(base)
        raw.update(experiment_id=EXP, seed=11, output_root=str(output))
        raw['data']['cache_dir'] = str(ROOT / 'cache' / EXP)
        raw['splits']['train'] = dict(start=w['train_start'], end=w['train_end_exclusive'])
        raw['splits']['validation'] = dict(start=w['evaluation_start'], end=w['evaluation_end_exclusive'])
        RunConfig.model_validate(raw)
        windows.append(w)
        configs[name] = raw
    registry = output / 'roles.json'
    write_json(registry, {'evaluation_year': 2022, 'reserved': ['2023-08-03/2023-08-05', '2024-07', '2024-Q4'],
                          'rule': 'Only 2021-12 training and 2022 train/evaluation; never evaluate before train end'})
    plan = dict(schema_version=1, experiment_id=EXP, windows=windows, configs=configs,
                registry_path=str(registry), registry_sha256=file_sha256(registry),
                recipe='seed11/raw12/all8/ten trading days/1500 steps/train_std/no smoothing',
                budget={'fits_max': 12, 'job_wall_seconds': 3600, 'total_wall_seconds': 28800},
                rationale='2022 is complete and avoids all existing reserved 2023/2024 periods',
                sources=['https://qlib.readthedocs.io/en/stable/component/report.html'])
    write_json(output / 'plan.json', plan)
    return plan


def worker(root, month, kind):
    plan = read_json(root / 'plan.json')
    w = next(w for w in plan['windows'] if w['name'] == month)
    validate_window(w)
    cfg = RunConfig.model_validate(plan['configs'][month])
    out = root / kind / month
    out.mkdir(parents=True, exist_ok=False)
    rec = Recorder(out, EXP)
    write_json(out / 'manifest.json', dict(status='running', run_id=f'{kind}-{month}', experiment_id=EXP,
               config=cfg.model_dump(mode='json'), code=capture_source(out), environment=environment()))
    rec.event('run', 'started', kind=kind)
    try:
        if kind == 'fit':
            fit_worker(plan, w, cfg, out, rec)
        else:
            from modified_tlob.adapter import TLOBAdapter
            paths = {'fixed': root / 'fit' / '202201' / 'model.pt',
                     'monthly': root / 'fit' / month / 'model.pt'}
            hashes = {a: file_sha256(p) for a, p in paths.items()}
            models = {a: TLOBAdapter.load(p, device='cuda', recorder=rec) for a, p in paths.items()}
            frames = {a: [] for a in paths}
            for i in range(0, len(w['evaluation_dates']), 5):
                days = w['evaluation_dates'][i:i+5]
                first, last = pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
                data = cfg.data.model_dump(mode='json') | {'labels_path': None}
                bundle = load_data(data, start=first, end=last)
                record_data(rec, out / f'data_{i}.json', bundle)
                if not bundle.labels.empty:
                    raise ValueError('Inference read labels')
                keys = inference_keys(bundle, Period(start=first, end=last), cfg.data)
                batch = build_batch(bundle.bars, keys, cfg.data)
                cutoff = batch.keys.date.to_numpy(dtype='datetime64[ns]')[:, None]
                if not (((batch.history_dates <= cutoff) | np.isnat(batch.history_dates)).all()
                        and batch.coverage['min_section_coverage'] >= cfg.acceptance.min_coverage):
                    raise ValueError('Causality/history coverage failed')
                for arm, model in models.items():
                    rec.event('inference', 'started', arm=arm, dates=days, labels_loaded=False)
                    frames[arm].append(batch.keys.assign(score=model.predict(batch)))
                    rec.event('inference', 'completed', arm=arm, resources=resource_usage())
                del batch, bundle
                gc.collect()
            for arm in paths:
                pd.concat(frames[arm]).to_parquet(out / f'{arm}.parquet', index=False)
            if hashes != {a: file_sha256(p) for a, p in paths.items()}:
                raise ValueError('Weights changed')
            write_json(out / 'weights.json', {a: {'path': str(p), 'sha256': hashes[a]} for a, p in paths.items()})
        rec.event('run', 'completed')
        finish(out, 'completed', 0)
    except BaseException as exc:
        rec.failure('run', exc)
        finish(out, 'failed', 1, error={'reason': str(exc)})
        raise


def analyze(root):
    plan = read_json(root / 'plan.json')
    targets = []
    for w in plan['windows']:
        cfg = RunConfig.model_validate(plan['configs'][w['name']])
        data = cfg.data.model_dump(mode='json')
        # Bounded standard-loader chunks preserve mapping and universe conventions.
        for i in range(0, len(w['evaluation_dates']), 5):
            days = w['evaluation_dates'][i:i+5]
            bundle = load_data(data, start=pd.Timestamp(days[0]),
                               end=pd.Timestamp(days[-1]) + pd.Timedelta(days=1))
            targets.append(select_targets(bundle.labels, cfg.splits.validation, sample_times=cfg.data.sample_times))
            del bundle
            gc.collect()
    target = pd.concat(targets, ignore_index=True)
    target.to_parquet(root / 'targets.parquet', index=False)
    all_sections, all_months, all_times = [], [], []
    calendar = sorted(target.date.dt.normalize().unique())
    for arm in ['fixed', 'monthly']:
        pred = pd.concat([pd.read_parquet(root / 'infer' / w['name'] / f'{arm}.parquet') for w in plan['windows']])
        coverage = _answer_coverage_check(pred[['date', 'instrument']], target, cfg.acceptance.min_coverage,
                                           'answer_coverage', 'instruments')
        if not coverage['passed']:
            raise ValueError('Answer coverage failed')
        scored = pred.merge(target[['date', 'instrument']], on=['date', 'instrument'], validate='one_to_one')
        metrics, daily, sections = evaluate(scored, target, cfg.evaluation, return_sections=True,
                                            undefined_constant_portfolios=True)
        sections['arm'] = arm
        ts = pd.to_datetime(sections.timestamp)
        sections['month'], sections['time'] = ts.dt.strftime('%Y-%m'), ts.dt.strftime('%H:%M')
        cutoff = {w['name'][:4]+'-'+w['name'][4:]: pd.Timestamp(w['train_dates'][-1]) for w in plan['windows']}
        last = pd.Series(pd.Timestamp(plan['windows'][0]['train_dates'][-1]), index=sections.index) if arm == 'fixed' else sections.month.map(cutoff)
        sections['weight_age_calendar_days'] = (ts.dt.normalize()-last).dt.days
        sections['weight_age_trading_days'] = [sum(pd.Timestamp(d) <= timestamp.normalize()
            and pd.Timestamp(d) > trained for d in calendar) for timestamp, trained in zip(ts, last)]
        if len(sections) != len(calendar)*8 or sections.groupby('date').size().ne(8).any():
            raise ValueError('Incomplete eight-time annual grid')
        cols = ['rank_ic', 'long_return', 'short_return', 'long_short_return', 'turnover', 'weight_age_calendar_days', 'weight_age_trading_days']
        for keys, destination in [(['arm','month'], all_months), (['arm','month','time'], all_times)]:
            grouped = sections.groupby(keys, sort=True)
            summary = grouped[cols].mean().join(grouped.size().rename('sections')).reset_index()
            destination.append(summary)
        all_sections.append(sections)
        daily.to_csv(root / f'{arm}_daily.csv', index=False)
        write_json(root / f'{arm}_metrics.json', metrics)
    pd.concat(all_sections).to_csv(root / 'by_timestamp.csv', index=False)
    pd.concat(all_months).to_csv(root / 'monthly.csv', index=False)
    pd.concat(all_times).to_csv(root / 'monthly_eight_times.csv', index=False)
    write_json(root / 'checks.json', {'passed': True, 'year': 2022, 'months_per_arm': 12, 'times_per_month': 8,
               'causal_weights': True, 'reserved_period_views': 0, 'scope': 'local approximate gross returns, no fees',
               'turnover': 'continuous across months, half L1 long-short weight difference; first section undefined'})


def run(root):
    plan = read_json(root / 'plan.json') if (root / 'plan.json').exists() else prepare(root)
    started = time.monotonic()
    for w in plan['windows']:
        for kind in ['fit', 'infer']:
            out = root / kind / w['name']
            if (out / 'manifest.json').exists():
                if read_json(out / 'manifest.json')['status'] == 'completed':
                    continue
                raise ValueError(f'Failed job requires investigation: {out}')
            if time.monotonic()-started > plan['budget']['total_wall_seconds']:
                raise TimeoutError('Annual budget exhausted')
            log = root / f'{kind}_{w["name"]}.log'
            with log.open('x', encoding='utf-8') as f:
                subprocess.run([sys.executable, '-m', 'scripts.annual_baseline', 'worker', str(root),
                                '--month', w['name'], '--kind', kind], stdout=f, stderr=f, check=True,
                               timeout=plan['budget']['job_wall_seconds'])
            print(f'{kind} {w["name"]} completed', flush=True)
    analyze(root)
    print('Annual diagnosis completed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'worker', 'analyze'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--month')
    parser.add_argument('--kind')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root.resolve(), args.month, args.kind)
    elif args.action == 'analyze':
        analyze(args.root.resolve())
    else:
        run(args.root.resolve())
