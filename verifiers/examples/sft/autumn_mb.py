import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 verifiers/examples/sft/wordle.py
"""

# convenience function for FA2 initialization
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", use_liger=False)
dataset = load_dataset('mokshjain/V0-autumn', split='train')

tok_counts = []
import pdb; pdb.set_trace();
for row in dataset:
    # count tokens in (prompt, completion)
    messages = row['prompt'][:-1] + row['completion'] # type: ignore
    toks = tokenizer.apply_chat_template( 
        messages,
        tokenize=True
    )
    tok_counts.append(len(toks))

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=8192,
    output_dir="sft-autumn",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    max_grad_norm=0.1,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=False,
    hub_model_id="Qwen2.5-0.5B-Autumn-SFT",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset # type: ignore
)
trainer.train()