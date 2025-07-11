import json

def aggregate_and_average(json_list):
    rewards = []
    judge_rewards = []
    format_rewards = []

    for entry in json_list:
        if 'reward' in entry:
            rewards.append(entry['reward'])
        if 'judge_reward_func' in entry:
            judge_rewards.append(entry['judge_reward_func'])
        if 'format_reward_func' in entry:
            format_rewards.append(entry['format_reward_func'])

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    print(rewards, judge_rewards, format_rewards)

    avg_reward = safe_avg(rewards)
    avg_judge_reward = safe_avg(judge_rewards)
    avg_format_reward = safe_avg(format_rewards)

    return {
        'avg_reward': avg_reward,
        'avg_judge_reward': avg_judge_reward,
        'avg_format_reward': avg_format_reward
    }

if __name__ == "__main__":
    json_file_path = "verifiers/textworld_eval/textworld_eval_qwen2b.json"  # <-- Change this to your actual JSON file path

    # Load the JSON list from file
    with open(json_file_path, 'r') as f:
        json_list = json.load(f)

    # Sanity check: Make sure the loaded JSON is a list
    if not isinstance(json_list, list):
        raise ValueError("Expected the JSON file to contain a list of JSON objects.")

    results = aggregate_and_average(json_list)
    print(results)
