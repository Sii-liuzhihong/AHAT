"""Compare real native dispatch tensors and FSDP checkpoints from two smoke runs."""
import argparse
import json
from pathlib import Path


def compare_tensors(left, right):
    import torch
    rows = {}
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            rows[key] = {'equal': False, 'reason': 'missing key'}
        elif a.shape != b.shape:
            rows[key] = {'equal': False, 'left_shape': list(a.shape), 'right_shape': list(b.shape)}
        else:
            rows[key] = {'equal': bool(torch.equal(a, b)), 'max_abs_difference': float((a.float() - b.float()).abs().max())}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-equal', action='store_true')
    args = parser.parse_args()
    import torch
    a, b = Path(args.left), Path(args.right)
    batch = compare_tensors(torch.load(a / f'native_batch_step_{args.step}.pt', weights_only=True),
                            torch.load(b / f'native_batch_step_{args.step}.pt', weights_only=True))
    checkpoint = Path('checkpoints') / f'global_step_{args.step}' / 'actor' / 'model_world_size_1_rank_0.pt'
    model = compare_tensors(torch.load(a / checkpoint, weights_only=False, map_location='cpu'),
                            torch.load(b / checkpoint, weights_only=False, map_location='cpu'))
    report = {'left': str(a), 'right': str(b), 'step': args.step,
              'batch_tensors': batch, 'model_tensors': model,
              'equal': all(row['equal'] for row in [*batch.values(), *model.values()])}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'equal': report['equal'], 'batch_tensors': len(batch), 'model_tensors': len(model)}, indent=2))
    if args.require_equal and not report['equal']:
        raise SystemExit('Native comparison differs; inspect the complete report')


if __name__ == '__main__':
    main()
