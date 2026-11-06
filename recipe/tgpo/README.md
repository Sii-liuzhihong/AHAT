# Subgoal TGPO recipe

The public input is canonical JSON/JSONL: `instruction`, `scene_graph`,
`domain_file_path`, and a supervised `output` (or `reply_list`) for SFT.
Relative domain paths resolve against the input file. Private data extraction
precedes this boundary. No original dataset is included.

From the installed repository:

```bash
python -m robot_planning.data.prepare_sft --input recipe/tgpo/examples/tasks.jsonl --output outputs/sft/train.parquet
python -m robot_planning.data.prepare_rl --input recipe/tgpo/examples/tasks.jsonl --output outputs/rl/train.parquet
```

Prepare an independently held-out validation file with the same commands.
Set `TGPO_MODEL`, `TGPO_TRAIN_DATA`, `TGPO_VAL_DATA`, and optionally
`TGPO_OUTPUT`, then run `bash recipe/tgpo/train_sft.sh` or
`bash recipe/tgpo/train_tgpo.sh`. `N_GPUS` defaults to 8. Additional arguments
are Hydra overrides. The TGPO configuration is a candidate reproduction of
one historical single-review, threshold-0.51 recipe. The author has not yet
selected a definitive main-result configuration. It uses 8 continuations and a tensor-parallel size of 2;
override these together when using fewer GPUs.
The configured 32768-token vLLM batch budget covers the 15000-token prompt
plus 3096-token response with chunked prefill disabled. Lowering the budget
below that combined length prevents vLLM startup.

SFT stores `prompt.question` and `answer.answer` and includes the original
`You are a helpful assistant.` system message. SFT joins the task prompt and
input with one newline; the original RL input uses two newlines and strips
outer whitespace. These differences are retained intentionally.
The original checkpoint-eval wrapper uses two newlines without outer stripping
and the system message `You are a helpful assistant` (without a final period).
Inference retains this third formatting variant so existing results remain comparable.

Repair/review use an OpenAI-compatible endpoint configured through
`OPENAI_BASE_URL` (default `http://localhost:8000/v1`), `OPENAI_API_KEY`
(optional for local servers), and `TGPO_LLM_MODEL`. No filesystem queue,
embedded credentials, or machine-specific path is required.
Set `TGPO_LLM_SEED` to pin the endpoint request seed when the server supports
it; this complements the local evaluation seed. Temperature zero alone did
not make every independent real-model request identical during validation.

```bash
bash recipe/tgpo/generate.sh --model /path/to/exported_hf_model --input recipe/tgpo/examples/tasks.jsonl --output outputs/predictions.jsonl
bash recipe/tgpo/evaluate.sh --help
```

Generation defaults match the source: temperature 0.7, top-p 0.9, and 3096
new tokens. `--n` retains every alternative in `reply_list`. Add
`--score-output outputs/scored.jsonl` to generate and score in one process.
Scoring existing predictions imports neither vLLM nor training dependencies.

The examples are hand-written API/solver smoke inputs and do not demonstrate
research performance. The tiny example domain is intentionally limited;
use your matching domain and task scene graphs for research runs.
