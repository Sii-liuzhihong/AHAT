"""Create a small random model for real GPU integration, not quality evaluation."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator([
        "system user assistant You are a helpful assistant. Move the robot to the kitchen.",
        "<task_decomposition> Move to kitchen. </task_decomposition> <think> inspect </think>",
        "<subgoal> (at robot kitchen) </subgoal> <related_objects> [robot,kitchen] </related_objects>",
        "<fixed_cot> Move to kitchen. </fixed_cot> Good Normal Bad plan response evaluate score",
    ], trainer)
    public_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        model_max_length=8192, padding_side="left",
    )
    public_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    public_tokenizer.save_pretrained(output)
    config = Qwen2Config(
        vocab_size=len(public_tokenizer), hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=8192, tie_word_embeddings=True,
        bos_token_id=public_tokenizer.bos_token_id, eos_token_id=public_tokenizer.eos_token_id,
        pad_token_id=public_tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(output, safe_serialization=True)
    print(f"Created {sum(p.numel() for p in model.parameters()):,} parameter model at {output}")


if __name__ == "__main__":
    main()
