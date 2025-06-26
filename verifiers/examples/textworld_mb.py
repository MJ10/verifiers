import verifiers as vf
from verifiers.envs.textworld_env import TextWorldEnv
from openai import OpenAI
import argparse
import random
from verifiers.rubrics import TextWorldRubric

import glob
from pathlib import Path
import textworld
import os

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
training_args.async_generation_timeout=1000

model, tokenizer = vf.get_model_and_tokenizer(args.model_name)

TEXTWORLD_PATH = "/Users/jean/Documents/verifiers/verifiers/tw_games"
tasks = ["the_cooking_game"]


def get_environment_ids(textworld_games_path, tasks, max_steps, seed=0):
    request_infos = textworld.EnvInfos(
        objective=True,
        description=True,
        score=True,
        max_score=True,
        won=True,
        admissible_commands=True
    )

    env_ids = []
    for pattern in ["*.ulx", "*.z8"]:
        for entry in sorted(glob.glob(os.path.join(textworld_games_path, f"**/{pattern}"), recursive=True)):
            task = Path(entry).parent.name
            if task in tasks:
                env_id = textworld.gym.register_game(entry, request_infos, max_episode_steps=max_steps)
                env_ids.append(env_id)

    return env_ids

env_ids = get_environment_ids(TEXTWORLD_PATH, tasks)
# randomly split into train and eval
train_env_ids = random.sample(env_ids, int(len(env_ids) * 0.8))
eval_env_ids = [env_id for env_id in env_ids if env_id not in train_env_ids]


vf_env = TextWorldEnv(
    programs_dir=args.programs_dir,
    data_dir=args.data_dir,
    train_list=["tw-v16"],
    eval_list=["tw-v16"],
    max_turns=10,
    seed=args.seed,
    std_lib_path=args.std_lib_path,
    rubric=TextWorldRubric()
)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

trainer.train()

