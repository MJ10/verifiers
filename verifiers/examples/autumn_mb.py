import verifiers as vf
from verifiers.envs.autumn_env import AutumnEnv, get_environment_ids
from openai import OpenAI
import argparse
import random

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 4

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/autumn_mb.py
"""

size = '0.5B'
model_name = f'Qwen/Qwen2.5-{size}-Instruct'
programs_dir = "programs/"
data_dir = "data/"

argparser = argparse.ArgumentParser()
argparser.add_argument("--model-name", type=str, default=model_name)
argparser.add_argument("--programs-dir", type=str, default="programs/")
argparser.add_argument("--data-dir", type=str, default="data/")
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--std-lib-path", type=str, default="autumn_stdlib.sexp")
args = argparser.parse_args()

run_name = f"autumn-grpo-{args.model_name}"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=6
training_args.num_generations=12
training_args.gradient_accumulation_steps=4
training_args.max_prompt_length=8192
training_args.max_completion_length=2048
training_args.max_steps=100
training_args.mask_env_responses=True

model, tokenizer = vf.get_model_and_tokenizer(args.model_name)

env_ids = get_environment_ids(args.programs_dir)
# randomly split into train and eval
train_env_ids = random.sample(env_ids, int(len(env_ids) * 0.8))
eval_env_ids = [env_id for env_id in env_ids if env_id not in train_env_ids]


vf_env = AutumnEnv(
    programs_dir=args.programs_dir,
    data_dir=args.data_dir,
    train_list=["ice", "grow"],
    eval_list=["wind"],
    max_turns=20,
    seed=args.seed,
    std_lib_path=args.std_lib_path,
    rubric=vf.rubrics.ModelBasedRubric(
        data_dir=args.data_dir,
        num_eval_transitions=10,
        judge_client=OpenAI(base_url=f"http://{args.vllm_server_host}:{args.vllm_server_port}/v1", api_key="abc"),
        judge_model=model_name,
    ),
)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

trainer.train()

