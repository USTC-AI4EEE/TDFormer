"""Evaluate prefixes of saved 48-hour forecasts without retraining or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ('full', 'wo_tan', 'wo_fan', 'wo_van')
LABELS = dict(full='Full', wo_tan='w/o TAN', wo_fan='w/o FAN', wo_van='w/o VAN')
PREFIXES = (1, 8, 16, 24, 48)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, predicted):
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)
    mask = actual > 0
    y, p = actual[mask], predicted[mask]
    assert len(y) > 1 and np.isfinite(y).all() and np.isfinite(p).all()
    error = y - p
    rmse = np.sqrt(np.mean(np.square(error)))
    return dict(
        mae=float(np.mean(np.abs(error))), rmse=float(rmse),
        positive_count=int(mask.sum()), total_count=len(actual),
    )


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=project / 'original_horizon_ablation_spring')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    config_path = Path(__file__).resolve().parents[1] / 'provenance' / 'experiment_config.json'
    config = json.loads(config_path.read_text())
    assert config['seed'] == args.seed and config['frequency'] == '1h'
    data = Path(config['data_path'])
    if not data.is_absolute():
        data = Path(__file__).resolve().parents[1] / data
    assert file_hash(data) == config['data_sha256']
    frame = pd.read_csv(data, encoding='gbk')
    frame = frame[frame['站点编号'].astype(str).str.lower().eq(config['station'].lower())].copy()
    frame['stamp'] = pd.to_datetime(frame['时间'])
    frame = frame.sort_values('stamp').reset_index(drop=True)
    assert frame['stamp'].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert frame['stamp'].dt.year.eq(2022).all() and frame['stamp'].dt.month.isin((3, 4, 5)).all()
    assert len(frame) == config['rows'] == 2208
    frame['power'] = pd.to_numeric(frame['出力(MW)'].replace('<NULL>', np.nan), errors='coerce')
    frame['power'] = frame['power'].interpolate(limit_direction='both').clip(lower=0)
    power = (frame['power'].to_numpy() / (config['capacity_mw'] + 1e-6)).astype(np.float32)
    input_length, trained_length = config['input_length'], 48
    assert input_length == 24
    test = frame.iloc[int(0.9 * len(frame)):].reset_index(drop=True)
    test_power = power[int(0.9 * len(frame)):]
    windows = len(test) - input_length - trained_length + 1
    actual = np.stack([test_power[i + input_length:i + input_length + trained_length]
                       for i in range(windows)])
    assert actual.shape == (150, 48)
    cumulative, by_lead, audits, payloads = [], [], [], []
    source_hashes = {}
    for variant in VARIANTS:
        source = root / 'horizon_48h' / 'metrics' / variant / f'seed_{args.seed}.json'
        item = json.loads(source.read_text())
        assert item['status'] == 'complete' and item['seed'] == args.seed
        assert item['variant'] == item['module_ablation'] == variant
        assert item['input_length'] == input_length and item['prediction_length'] == trained_length
        assert item['test_window_count'] == windows and item['data_path'] == str(data)
        assert item['model_hyperparameters'] == config['model_hyperparameters']
        prediction_path = Path(item['saved_prediction_path'])
        prediction = np.load(prediction_path, allow_pickle=False)
        assert prediction.shape == (windows * trained_length,) and np.isfinite(prediction).all()
        prediction = prediction.reshape(windows, trained_length)
        full_metrics = metrics(actual, prediction)
        assert full_metrics['positive_count'] == item['evaluation_positive_count']
        for name, expected in item['metrics'].items():
            difference = abs(full_metrics[name] - expected)
            assert difference < 1e-10, (variant, name, difference)
            audits.append(dict(variant=variant, metric=name, stored=expected,
                               reconstructed=full_metrics[name], absolute_difference=difference))
        source_hashes[str(source)] = file_hash(source)
        source_hashes[str(prediction_path)] = file_hash(prediction_path)
        payloads.append(item)
        for prefix in PREFIXES:
            row = dict(seed=args.seed, trained_prediction_hours=48, prefix_hours=prefix,
                       variant=variant, test_windows=windows)
            row.update(metrics(actual[:, :prefix], prediction[:, :prefix]))
            cumulative.append(row)
        for lead in range(48):
            row = dict(seed=args.seed, lead_hour=lead + 1, variant=variant, test_windows=windows)
            row.update(metrics(actual[:, lead], prediction[:, lead]))
            by_lead.append(row)
    assert len({item['initialization_audit']['reference_full_sha256'] for item in payloads}) == 1
    assert len({item['initialization_audit']['shared_trunk_sha256'] for item in payloads}) == 1
    result = pd.DataFrame(cumulative)
    for prefix in PREFIXES:
        group = result[result.prefix_hours.eq(prefix)]
        assert group.positive_count.nunique() == group.total_count.nunique() == 1
    deltas = []
    for prefix in PREFIXES:
        group = result[result.prefix_hours.eq(prefix)].set_index('variant')
        for variant in VARIANTS[1:]:
            deltas.append(dict(
                prefix_hours=prefix, ablation=variant,
                mae_ablation_minus_full_pp=(group.loc[variant, 'mae'] - group.loc['full', 'mae']) * 100,
                rmse_ablation_minus_full_pp=(group.loc[variant, 'rmse'] - group.loc['full', 'rmse']) * 100,
            ))
    out = root / f'prefix_evaluation_48h_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'prefix_metrics.csv', index=False)
    pd.DataFrame(by_lead).to_csv(out / 'individual_lead_metrics.csv', index=False)
    pd.DataFrame(deltas).to_csv(out / 'ablation_deltas.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'full_48h_reproduction_audit.csv', index=False)
    window_rows = []
    for index in range(windows):
        row = dict(window=index, last_history_timestamp=str(test.iloc[index + input_length - 1]['stamp']))
        for prefix in PREFIXES:
            row[f'prediction_start'] = str(test.iloc[index + input_length]['stamp'])
            row[f'prefix_{prefix}h_end'] = str(test.iloc[index + input_length + prefix - 1]['stamp'])
        window_rows.append(row)
    pd.DataFrame(window_rows).to_csv(out / 'test_window_timestamps.csv', index=False)
    manifest = dict(
        seed=args.seed, history_hours=24, trained_prediction_hours=48,
        cumulative_prefix_hours=[1, 8, 16, 24], reference_full_hours=48,
        windows=windows, input_csv=str(data), input_csv_sha256=file_hash(data),
        data_frequency='1h', capacity_mw=config['capacity_mw'],
        evaluation='Keep the same 150 origins; take prediction[:, :H] and truth[:, :H]; flatten; evaluate only truth>0',
        training_or_inference_performed=False, source_sha256=source_hashes,
        evaluator_sha256=file_hash(__file__),
    )
    (out / 'evaluation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = [
        f'# seed={args.seed}：48h预测模型的前缀评估', '',
        '本次只读取已保存的48h模型预测，不训练、不重新推理。固定相同的150个测试起报窗口，分别取每个48步输出的前1、8、16、24步进行累计评估；另外列出完整48h作参考。', '',
        '历史输入24h，春季1h数据。四个结构沿用原实验参数及最终保存checkpoint的输出。', '',
        '## MAE / RMSE对比', '',
        '每格为MAE / RMSE（容量归一化后乘100，%），越低越好。前H小时指第1到第H步合并，不是只评估第H步。', '',
        '| 评估区间 | Full | w/o TAN | w/o FAN | w/o VAN | 正功率评估点 / 总点数 |',
        '|---|---|---|---|---|---|',
    ]
    for prefix in PREFIXES:
        group = result[result.prefix_hours.eq(prefix)].set_index('variant')
        cells = [f"{group.loc[v, 'mae']*100:.3f} / {group.loc[v, 'rmse']*100:.3f}" for v in VARIANTS]
        count = group.iloc[0]
        lines.append(f'| 前{prefix}h | ' + ' | '.join(cells)
                     + f' | {int(count.positive_count)} / {int(count.total_count)} |')
    lines.extend(['', '## 观察', ''])
    for prefix in PREFIXES:
        group = result[result.prefix_hours.eq(prefix)]
        best_mae = group.loc[group.mae.idxmin()]
        best_rmse = group.loc[group.rmse.idxmin()]
        lines.append(f'- 前{prefix}h：MAE最低为{LABELS[best_mae.variant]}（{best_mae.mae*100:.3f}%）；'
                     f'RMSE最低为{LABELS[best_rmse.variant]}（{best_rmse.rmse*100:.3f}%）。')
    lines.extend(['', '## 解释边界', '',
                  '- 这是48h模型输出的截取评估，不是分别训练1/8/16/24h模型',
                  '- prefix_metrics.csv含MAE和RMSE；individual_lead_metrics.csv按单个提前量逐步统计；ablation_deltas.csv列消融−Full误差差值；full_48h_reproduction_audit.csv为完整48h复算审计。'])
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('AUDIT PASS: all 4 original 48h metric sets reproduced; 150 aligned windows; no training/inference.')
    print(result[['prefix_hours', 'variant', 'mae', 'rmse', 'positive_count']].to_string(index=False))
    print('Report:', out / 'report.md')


if __name__ == '__main__':
    main()
