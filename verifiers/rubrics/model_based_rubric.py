import os
from openai import OpenAI
import random
import json
from typing import List, Dict
from verifiers.parsers import Parser
from verifiers.rubrics.rubric import Rubric
import numpy as np

NSP_JUDGE_PROMPT = """Given a model, an observation, and an action, \
produce the next state of the environment.

Model:
```
{model}
```

Observation:
```
{observation}
```

Action:
```
{action}
```

Produce the next state of the environment."""


AP_JUDGE_PROMPT = """Given a model, an observation, and the next observation, \
produce the action that leads to the next observation.

Model:
```
{model}
```

Observation:
```
{observation}
```

Next Observation:
```
{next_observation}
```

Produce the action that leads to the next observation."""

class ModelBasedRubric(Rubric):
    def __init__(self,
                 judge_client: OpenAI | None = None,
                 judge_model: str = "gpt-4.1-nano",
                 judge_prompt: str = NSP_JUDGE_PROMPT,
                 parser: Parser = Parser(),
                 data_dir: str = "data/",
                 num_eval_transitions: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.parser = parser
        self.data_dir = data_dir
        self.num_eval_transitions = num_eval_transitions
        self.add_reward_func(self.judge_reward_func)

    def load_transitions(self, env_name: str) -> List[Dict]:
        with open(f"{self.data_dir}/{env_name}.json", "r") as f:
            trajectory = json.load(f)
        # The trajectory has key "observations" whcih is a list of steps where each step is a dict with keys:
        # "action" and "rendered_output". We want to extract the transitions between the steps.
        transitions = []
        for i in range(len(trajectory['observations']) - 1):
            transitions.append({
                'observation': trajectory['observations'][i]["rendered_output"],
                'action': trajectory['observations'][i]['action'],
                'next_observation': trajectory['observations'][i+1]["rendered_output"]
            })
        return transitions

    def judge_reward_func(self, prompt, state, **kwargs) -> float:
        # prompt contains name of the env
        
        model = state['model']
        transitions = self.load_transitions(prompt)

        eval_transitions = random.sample(transitions, self.num_eval_transitions)
        scores = []
        for transition in eval_transitions:
            observation = transition['observation']
            action = transition['action']
            next_observation = transition['next_observation']
            nsp = random.random() < 0.5
            if nsp:
                prompt = NSP_JUDGE_PROMPT.format(model=model, observation=observation, action=action)
            else:
                prompt = AP_JUDGE_PROMPT.format(model=model, observation=observation, next_observation=next_observation)
            
            judge_response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
            )
            judge_response = str(judge_response.choices[0].message.content)
            if nsp:
                score = judge_response == next_observation
            else:
                score = judge_response == action
            scores.append(score)
            
        
        return np.mean(scores)
    


    