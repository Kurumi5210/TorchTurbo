"""Compare the token arrays printed in DPO logs using only the standard library."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


EMPTY_THINK = [248068, 271, 248069, 271]


@dataclass
class LoggedSequence:
    input_ids: list[int]
    labels: list[int]
    input_line: int
    labels_line: int


def read_records(path: Path, is_csv: bool) -> list[tuple[int, str]]:
    """Keep LF source line numbers even when progress bars contain CR."""
    raw = path.read_bytes().decode('utf-8')
    if not is_csv:
        return list(enumerate(raw.split('\n'), 1))
    pieces = raw.split('\n')
    lines = [part + '\n' for part in pieces[:-1]]
    if pieces[-1]:
        lines.append(pieces[-1])
    csv.field_size_limit(50_000_000)
    reader = csv.reader(iter(lines))
    assert next(reader) == ['_log']
    previous_end = reader.line_num
    records = []
    for record in reader:
        start = previous_end + 1
        previous_end = reader.line_num
        assert len(record) == 1
        content = record[0].split(', content=', 1)[1].rsplit(', __pack_shard_id__=', 1)[0]
        records.append((start, content))
    return records


def extract_sequences(records: list[tuple[int, str]]) -> dict[str, LoggedSequence]:
    sequences = {}
    for branch in ['chosen', 'rejected']:
        fields = {}
        for field in ['input', 'labels']:
            prefix = f'[INFO:swift] [{branch.upper()}_{field.upper()}_IDS] '
            hits = [(line, text[len(prefix):]) for line, text in records if text.startswith(prefix)]
            assert len(hits) == 1, (branch, field, len(hits))
            line, value = hits[0]
            fields[field] = (line, ast.literal_eval(value))
        sequence = LoggedSequence(fields['input'][1], fields['labels'][1], fields['input'][0], fields['labels'][0])
        assert len(sequence.input_ids) == len(sequence.labels)
        assert all(label in {-100, token} for token, label in zip(sequence.input_ids, sequence.labels))
        sequences[branch] = sequence
    return sequences


def contiguous_ranges(indices: list[int]) -> list[list[int]]:
    """Return zero-based, half-open ranges."""
    result = []
    for index in indices:
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result


def sequence_summary(sequence: LoggedSequence) -> dict:
    supervised = [i for i, value in enumerate(sequence.labels) if value != -100]
    return {
        'input_source_line': sequence.input_line,
        'labels_source_line': sequence.labels_line,
        'num_input_tokens': len(sequence.input_ids),
        'num_supervised_tokens': len(supervised),
        'supervised_ranges': contiguous_ranges(supervised),
    }


def compare_branch(gpu: LoggedSequence, npu: LoggedSequence) -> tuple[dict, list[int]]:
    """Check exact equality after removing masked GPU empty-think blocks."""
    i = j = 0
    removed = []
    aligned_gpu_labels = []
    while i < len(gpu.input_ids) and j < len(npu.input_ids):
        if gpu.input_ids[i] == npu.input_ids[j]:
            aligned_gpu_labels.append(gpu.labels[i])
            i += 1
            j += 1
            continue
        assert gpu.input_ids[i:i + 4] == EMPTY_THINK, ('unexpected input difference', i, j)
        assert gpu.labels[i:i + 4] == [-100] * 4
        removed.append({'gpu_range': [i, i + 4], 'npu_position': j, 'token_ids': EMPTY_THINK})
        i += 4
    assert i == len(gpu.input_ids) and j == len(npu.input_ids)
    extra_npu = [i for i, (g, n) in enumerate(zip(aligned_gpu_labels, npu.labels)) if g == -100 and n != -100]
    extra_gpu = [i for i, (g, n) in enumerate(zip(aligned_gpu_labels, npu.labels)) if g != -100 and n == -100]
    supervised_gpu = [value for value in gpu.labels if value != -100]
    npu_runs = contiguous_ranges([i for i, value in enumerate(npu.labels) if value != -100])
    final_start, final_end = npu_runs[-1]
    return {
        'gpu': sequence_summary(gpu),
        'npu': sequence_summary(npu),
        'removed_gpu_empty_think_blocks': removed,
        'all_remaining_input_tokens_identical': True,
        'extra_npu_supervised_tokens': len(extra_npu),
        'extra_npu_supervised_ranges': contiguous_ranges(extra_npu),
        'extra_gpu_supervised_tokens': len(extra_gpu),
        'npu_final_supervised_span_equals_all_gpu_supervision': supervised_gpu == npu.labels[final_start:final_end],
    }, extra_npu


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-log', type=Path, default=root / 'gpu_kubemaker_log.csv')
    parser.add_argument('--npu-log', type=Path, default=root / 'log-0.txt')
    args = parser.parse_args()
    gpu = extract_sequences(read_records(args.gpu_log, True))
    npu = extract_sequences(read_records(args.npu_log, False))
    result = {'range_convention': 'zero-based, half-open', 'log_sha256': {}}
    for side, path in [('gpu', args.gpu_log), ('npu', args.npu_log)]:
        result['log_sha256'][side] = hashlib.sha256(path.read_bytes()).hexdigest()
    extra = {}
    for branch in ['chosen', 'rejected']:
        result[branch], extra[branch] = compare_branch(gpu[branch], npu[branch])
    assert extra['chosen'] == extra['rejected']
    result['extra_supervised_tokens_identical_between_chosen_and_rejected'] = all(
        npu['chosen'].input_ids[i] == npu['rejected'].input_ids[i] for i in extra['chosen'])
    last_shared = max(extra['chosen']) + 1 if extra['chosen'] else 0
    result['input_prefix_through_extra_supervision_identical_between_chosen_and_rejected'] = (
        npu['chosen'].input_ids[:last_shared] == npu['rejected'].input_ids[:last_shared])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
