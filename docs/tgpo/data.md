# Public data contract

The public input starts **after private source extraction**. Supply UTF-8 JSONL
or a JSON array with these fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable task identifier; generated from row index when omitted in inference |
| `instruction` | Nonempty natural-language task |
| `scene_graph` | Object or its JSON encoding: rooms, furnitures, objects, agent, links |
| `domain_file_path` | PDDL domain; relative paths resolve against the input file |
| `output` | Supervised response for SFT; `reply` or first `reply_list` is also accepted |
| `reply_list` | One or more generated responses when evaluating existing predictions |
| `dynamic_domain` | Optional PDDL text for the explicit dynamic-domain prompt variant |
| `dynamic_domain_path` | Alternative dynamic prompt PDDL file; relative to the input file |

The hand-written examples under `recipe/tgpo/examples/` describe a single
table-cleaning task. Links are `[parent, child, relation]` triples; names and
predicates must match the supplied domain. They illustrate the schema and
executable path, not the private benchmark distribution. Use an independent
validation split for experiments.

`prepare_sft` writes nested `prompt.question` and `answer.answer`, with
`data_source`. The recipe selects those keys in the existing verl SFTDataset.
`prepare_rl` writes `prompt` chat messages, `data_source`, `ability`,
`reward_model` (`style`, `ground_truth`) and `extra_info` (`split`, `index`,
`instruction`, serialized `scene_graph`, resolved `domain_file_path`). The
training loop exposes task metadata from that boundary for refinement.

## Prompt compatibility

Three historical paths differ, and those differences are preserved explicitly:

| Path | System message | User formatting |
| --- | --- | --- |
| SFT | `You are a helpful assistant.` | Task prompt + one newline + XML-tagged task/scene |
| TGPO/RL | `You are a helpful assistant.` | Two newlines between task prompt/input, outer whitespace stripped |
| Offline generation | `You are a helpful assistant` | Two newlines, outer whitespace retained |

The scene uses the original Python mapping representation inside the task
prompt. Replacing it with JSON would change the token sequence. Versioned
training, repair and reviewer text is packaged as resources; it can be loaded
after wheel installation and from an arbitrary working directory. Tests check
prompt bytes and real tokenizer tensors against the source.

Selecting `subgoal_system_prompt_with_predicate_dynamic_domain` requires caller-
provided `dynamic_domain` text or `dynamic_domain_path`. It uses the original
`.format(dynamic_domain=...)` substitution and rejects a missing domain instead
of training on a literal placeholder. Private random domain augmentation is
outside the public input boundary; it is not silently recreated.

The inference output preserves task context and all alternatives in `reply_list`,
plus generation parameters. Offline evaluation explicitly chooses `first` (the
historical script protocol) or `all` candidates. It does not claim pass@k.
