import json
import re
import os
import random

import numpy as np

from datasets import Dataset
from contextlib import contextmanager
from copy import deepcopy
from openai import OpenAI
from typing import Dict, Tuple, Any, List, Union
from transformers import PreTrainedTokenizerBase

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import ModelBasedRubric

import textworld 
import gym 
import textworld.gym


@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def get_environment_ids(base_dir: str):
    pass


def parse_grid(render_output: str):
    pass


def render_grid(grid: Dict[str, Any]) -> str:
    pass

def render_grid_numbers(grid: Dict[str, Any]) -> str:
    pass

SYS_PROMPT_ALT = """You are an AI agent tasked with exploring and understanding a grid-based environment. Your goal is to interact with the environment efficiently and effectively, trying to deduce the underlying rules that govern it. You will be provided with observations from the environment and your current model of understanding. Based on these, you should think, update your model if necessary, and decide on an action to take.

Here is the current observation of the environment:
<environment_observation>
{{ENVIRONMENT_OBSERVATION}}
</environment_observation>

Here is your current model of understanding:
<current_model>
{{CURRENT_MODEL}}
</current_model>

For each turn, follow these steps:

1. Think: Analyze the current observation and your existing model. Consider what you've learned so far and what you still need to explore or confirm. Think about the most effective action to take next. Do this inside <think> tags.

2. Update Model: Based on your thinking, update your model of understanding if necessary. If you're adding new information or modifying existing information, do this inside <model_edit> tags. If no update is needed, you can skip this step.

3. Choose Action: Decide on the next action to take. Your options are:
   * left: press the left arrow key
   * right: press the right arrow key
   * up: press the up arrow key
   * down: press the down arrow key
   * click <x> <y>: click on the cell at location (<x>, <y>)
   * NOP: do nothing
   * quit: quit the environment

   Place your chosen action inside <action> tags.

Your output should strictly follow this format:

<think>
[Your step-by-step thinking process]
</think>

<model_edit>
[Your updates to the model, if any]
</model_edit>

<action>
[Your chosen action]
</action>

Remember, your final output should only include these three elements: the thinking process, model edits (if any), and the chosen action. Do not include any additional text or explanations outside of these tags.
"""


SYSTEM_PROMPT = """You are an expert agent playing a text-based game. 
You will be given observations and available actions to choose from at each step. 
Your task is to interact with the environment efficiently to collect as many points as possible and win the game.
Your understanding of the environment should be captured in the model text. 

In each turn, think step-by-step inside <think>...</think> tags, generate the model in <model_edit>...</model_edit> tags, produce the action inside <action>...</action> tags.
Follow the format strictly."""


class TextWorldEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """
    def __init__(self,
                 programs_dir: str = "n/a",
                 seed: int = 0,
                 max_turns: int = 40,
                 rubric: ModelBasedRubric = None,
                 std_lib_path: str = "n/a",
                 train_list: List[str] = [],
                 eval_list: List[str] = [],
                 **kwargs):
        self.programs_dir = programs_dir
        self.seed = seed
        self.std_lib_path = std_lib_path
        self.render_mode = "json"
        parser = XMLParser(fields=["think", "model_edit" ,"action"], answer_field="action")
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

        self.dataset, self.eval_dataset = self.textworld_to_hf(train_list, eval_list)
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            max_turns=max_turns,
            dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            **kwargs
        )

        self.parser = parser
        self.rubric = rubric

    def is_completed(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        if 'is_finished' in state and state['is_finished']:
            state.pop('initialized')
            return state['is_finished']
        return False

    def render(self, render_dict: Dict[str, Any], error_msg: str = None):
        pass
        

    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # load env 
        if 'initialized' not in state or not state['initialized']:
            env_name = state['env_name']
            # env_id = textworld.gym.register_game(env_name,
            #                          max_episode_steps=10)
            env = textworld.gym.make(state['env_name']) 
            env = TextWorldWrapper(env, max_steps=self.max_turns)
            obsv = env.reset() 

            state['env'] = env 

            state['reward'] = 0.0

            state['max_reward'] = state['env'].get_max_score()

            available_actions = state['env'].get_available_actions()

            state['available_actions'] = available_actions

            self.is_terminal = False 
            state['initialized'] = True 

            print("~~~~~~~~~~~~")
            print(obsv)
            env_message = {"role" : "user", "content": obsv} 
            return env_message, state 

        
        # parse guess
        turn = self.parser.parse(messages[-1]["content"])
        action = str(turn.action)
        model_edit = turn.model_edit
        think = turn.think
        error_msg = None
        if action == "quit":
            self.is_terminal = True
        
        if action in state['env'].language_action_space:
            action = action 
        else: 
            # default action = help 
            action = state['env'].default_action()
            error_msg = "Invalid action: executing 'help' instead."

        obsv, reward, done, info = state['env'].step(action)
        state['reward'] = reward 
        self.is_terminal = done 

        state['model'] = model_edit
        state['is_finished'] = self.is_terminal

        next_available_actions = state['env'].get_available_actions()
        next_available_actions = "\n".join(f"- {a}" for a in next_available_actions) 

        obsv = obsv + f"\n\nAvailable actions:\n{next_available_actions}"

        if error_msg: obsv = error_msg + obsv

        env_message = {"role": "user", "content": obsv}
        return env_message, state

    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False

        completion = []
        turn = 0

       # import pdb; pdb.set_trace();
        state = {'model': "", 'answer': answer, "env_name": prompt[-1]["content"]}
        print(state['env_name'])

        # initialize env and get list of available actions
        env_msg, state = self.env_response([], state, **kwargs)
        available_actions = state.get('available_actions', []) 
        available_actions =  "\n".join(f"- {a}" for a in available_actions) if available_actions else "No actions available."
        self.system_prompt = self.system_prompt.format(
            available_actions="\n".join(available_actions)
        )

        messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

        messages.append(env_msg)
        completion.append(env_msg)

        response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
        has_error = response.startswith("[ERROR]")
        messages.append({"role": "assistant", "content": response})
        completion.append({"role": "assistant", "content": response})
        turn += 1

        if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
            is_completed = True

        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break

            env_msg, state = self.env_response(messages, state, **kwargs)
            messages.append(env_msg)
            completion.append(env_msg)

            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
        state['turn'] = turn
        return completion, state

    def textworld_to_hf(self, train_list, eval_list) -> Tuple[Dataset, Dataset]:
        dataset_rows = []
        eval_dataset_rows = []
        random.seed(self.seed)
        for env_name in train_list:
            dataset_rows.append({
                "question": env_name,
                "answer": env_name
            })
        
        for env_name in eval_list:
            eval_dataset_rows.append({
                "question": env_name,
                "answer": env_name
            })
        
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)
        return dataset.repeat(1), eval_dataset.repeat(1)
    
    def process_chat_format(
        self,
        prompt: List[Dict[str, str]],
        completion: List[Dict[str, str]],
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool = False
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Process chat format conversations using incremental prefixes.
        
        Logic:
        1. For each step, tokenize conversation prefix (prompt + completion[:i])
        2. Calculate token differences between steps to get individual message tokens
        3. Apply masking for intermediate responses if needed
        
        Returns:
            prompt_ids, prompt_mask, completion_ids, completion_mask
        """
        # tokenize just the prompt
        prompt = prompt[:-1] + [completion[0]]
        completion = completion[1:]
        prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        assert isinstance(prompt_text, str)
        prompt_ids = processing_class.encode(prompt_text)
        prompt_mask = [1] * len(prompt_ids)
        
        # track completion tokens and masks by processing incrementally
        completion_ids = []
        completion_mask = []
        
        # previous tokenization (starts with just prompt)
        prev_ids = prompt_ids
        
        # process each completion message incrementally
        for i, msg in enumerate(completion):
            # create conversation prefix: prompt + completion[:i+1]
            conversation_prefix = prompt + completion[:i+1]
            
            # tokenize the full prefix
            prefix_text = processing_class.apply_chat_template(
                conversation_prefix, 
                tokenize=False, 
                add_generation_prompt=False,
            )
            assert isinstance(prefix_text, str), f"Expected string from apply_chat_template, got {type(prefix_text)}"
            current_ids = processing_class.encode(prefix_text)
            assert current_ids[:len(prev_ids)] == prev_ids, f"Tokenization difference in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
            
            # add new tokens to completion tokens
            new_tokens = current_ids[len(prev_ids):] 
            assert len(new_tokens) > 0, f"No new tokens in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
            completion_ids.extend(new_tokens)

            # create mask
            if msg["role"] == "assistant":
                msg_mask = [1] * len(new_tokens)
            elif msg["role"] != "assistant" and mask_env_responses:
                # mask intermediate 'user' and/or 'tool' messages 
                msg_mask = [0] * len(new_tokens)
            else:
                # default to not masking
                msg_mask = [1] * len(new_tokens)
            
            completion_mask.extend(msg_mask)
            # Update previous tokenization for next iteration
            prev_ids = current_ids
            assert len(completion_ids) == len(completion_mask), f"Length mismatch in chat format. Completion ids: {completion_ids}, completion mask: {completion_mask}"

        return prompt_ids, prompt_mask, completion_ids, completion_mask
    

class AlwaysTrue:
    def __contains__(self, item):
        return True

class TextWorldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=40):
        super().__init__(env)
        self.language_action_space = AlwaysTrue()
        self.progression = 0.0
        self.max_steps = max_steps
        self.action_space = gym.spaces.Space()
        self.observation_space = gym.spaces.Space()
        self.last_info = None

    @property
    def default_action(self):
        return "help"

    def get_text_action(self, action):
        return action

    def textworld_process_obsv(self, textworld_obsv):
        return textworld_obsv 

    def filter_objective(self, obs, info):
        objective = info["objective"]
        parts = obs.split(objective)
        if len(parts) == 1:
            return parts[0].strip()
        else:
            return parts[-1].strip()

    def reset(self):
        obs, info = self.env.reset()
        obs = self.filter_objective(obs, info)
        self.progression = 0.0
        self.last_info = info

        return self.textworld_process_obsv(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.filter_objective(obs, info)
        self.last_info = info

        if done:
            self.progression = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)

        return self.textworld_process_obsv(obs), reward, done, info

    def get_stats(self):
        return {"progression": self.progression}
    
    def get_available_actions(self):
        if self.last_info and "admissible_commands" in self.last_info:
            return self.last_info["admissible_commands"]
        else:
            return []
    def get_max_score(self):
        if self.last_info and "max_score" in self.last_info:
            return self.last_info["max_score"]
        else:
            return 0.0