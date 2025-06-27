import os
from openai import OpenAI
import random

import verifiers as vf
from verifiers.envs.textworld_env import TextWorldEnv
from verifiers.rubrics import TextWorldRubric

import textworld
import glob
from pathlib import Path

TEXTWORLD_PATH = "./tw_games"
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


def main(api: str, num_samples: int, max_tokens: int, save_dataset: bool = False,
         data_dir: str = "data/", programs_dir: str = "tw_games/", seed: int = 0, std_lib_path: str = "n/a"):
    # collect V3/R1 rollouts from API
    if api == "deepseek":
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model_name = "deepseek-chat" # DeepSeek V3-0324
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "openai":
        # just for testing :) not for distillation :)
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4.1" 
        client = OpenAI(api_key=api_key)
    elif api == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = "google/gemini-2.5-flash-preview-05-20"
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif api == "local":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        client = OpenAI(api_key="abc", base_url="http://localhost:8000/v1")
    else:
        raise ValueError(f"Invalid API: {api}")
    sampling_args = {
        "max_tokens": max_tokens,
    }
    # env_ids = get_environment_ids(programs_dir)
    # randomly split into train and eval
    all_env_ids = get_environment_ids(TEXTWORLD_PATH, tasks, max_steps = 10, seed=seed)#[:20]
    random.seed(seed)
    random.shuffle(all_env_ids)
    split_idx = 1#int(len(all_env_ids))
    train_env_ids = all_env_ids[:split_idx]
    eval_env_ids = all_env_ids[split_idx:]
    print(len(train_env_ids), "train environments")
    print(len(eval_env_ids), "eval environments")

    vf_env = TextWorldEnv(
        programs_dir=programs_dir,
        data_dir=data_dir,
        train_list=train_env_ids,
        eval_list=eval_env_ids,
        max_turns = 10,
        seed=seed,
        std_lib_path=std_lib_path,
        rubric=TextWorldRubric(),
    )
    # vf_env.rollout()
    columns = ['prompt', 'completion', 'answer', 'reward', ...]
    # use deepseek-chat for multiturn rollouts (V3-0324)
    results = vf_env.evaluate(
        client=client,
        model=model_name, 
        sampling_args=sampling_args,
        num_samples=num_samples
    )

    print('--- Example ---')
    print('Prompt: ', results['prompt'][0])
    print('Completion: ', results['completion'][0])
    print('Answer: ', results['answer'][0])
    print("--- Rewards ---")
    for k, v in results.items():
        if 'reward' in k:
            print(k, '-', sum(v) / len(v)) 
    if save_dataset:
        dataset_dsv3 = vf_env.make_dataset(results)
        # filter to top half of rows by rewards
        dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True)
        # save to hub
        dataset_dsv3.push_to_hub("V0-textworld")

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--api", "-a", type=str, default="openai")
    argparser.add_argument("--num-samples", "-n", type=int, default=20)
    argparser.add_argument("--max-tokens", "-t", type=int, default=2048)
    argparser.add_argument("--save-dataset", "-s", action="store_true", default=False)
    argparser.add_argument("--data-dir", "-d", type=str, default="data/")
    argparser.add_argument("--programs-dir", "-p", type=str, default="programs/")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--std-lib-path", "-l", type=str, default="autumn_stdlib.sexp")
    args = argparser.parse_args()
    main(args.api, args.num_samples, args.max_tokens, args.save_dataset, args.data_dir, args.programs_dir, args.seed, args.std_lib_path)