"""Deterministic external-task fixtures for native worker smoke tests only.

These scores deliberately exercise TGPO selection/replacement. They do not
measure task quality and must not be used for reported research results.
"""

def repair_batch_records(records, **kwargs):
    return [{'fix_cot': 'Move to kitchen.', 'response': '<fixed_cot>Move to kitchen.</fixed_cot>'} for _ in records]


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    # Candidate responses are reconstructed with this legacy prefix.
    results = []
    for index, reply in enumerate(solution_strs):
        candidate = reply.startswith('<think>')
        score = [.75, 1.0][index % 2] if candidate else [.0, .25][index % 2]
        results.append({'score': score, 'evaluate': 'validation fixture feedback',
                        'reply': reply, 'reward_validate': int(candidate)})
    return results
