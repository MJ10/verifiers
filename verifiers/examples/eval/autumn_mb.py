import os
from openai import OpenAI
import random

import verifiers as vf
from verifiers.envs.autumn_env import AutumnEnv, get_environment_ids
from verifiers.rubrics import ModelBasedRubric


def main(api: str, num_samples: int, max_tokens: int, save_dataset: bool = False,
         data_dir: str = "data/", programs_dir: str = "programs/", seed: int = 0, std_lib_path: str = "autumn_stdlib.sexp"):
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
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        client = OpenAI(api_key="abc", base_url="http://localhost:8000")
    else:
        raise ValueError(f"Invalid API: {api}")
    sampling_args = {
        "max_tokens": max_tokens,
    }
    env_ids = get_environment_ids(programs_dir)
    # randomly split into train and eval
    train_env_ids = random.sample(env_ids, int(len(env_ids) * 0.8))
    eval_env_ids = [env_id for env_id in env_ids if env_id not in train_env_ids]

    vf_env = AutumnEnv(
        programs_dir=programs_dir,
        data_dir=data_dir,
        train_list=train_env_ids,
        eval_list=eval_env_ids,
        max_turns=50,
        seed=seed,
        std_lib_path=std_lib_path,
        rubric=ModelBasedRubric(
            data_dir=data_dir,
            num_eval_transitions=20,
            judge_client=client,
            judge_model=model_name,
            disable_rub=True
        ),
    )
    # columns = ['prompt', 'completion', 'answer', 'reward', ...]
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
        dataset_dsv3.push_to_hub("V0-autumn")

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