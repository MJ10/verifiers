from verifiers.envs import interpreter_module
import json
import re
import os
import random
from diff_match_patch import diff_match_patch
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

import gymnasium as gym
import minihack
from nle_language_wrapper import NLELanguageWrapper


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



SYSTEM_PROMPT = """You are a curious agent exploring a grid-based environment. 
You will be given observations as 
1) a description of the surroundings
2) the current message from the environment
3) your stats
4) the description of the cell under the current cursor,
and a lit of available actions to choose from at each step. 
Your task is to interact with the environment efficiently and effectively and try to understand the underlying rules of the environment.
Your understanding of the environment should be captured in the model text. 

The actions available are the following:
{available_actions}

In each turn, think step-by-step inside <think>...</think> tags, generate the model in <model_edit>...</model_edit> tags, produce the action inside <action>...</action> tags.
Follow the format strictly."""


class MiniHackWrapperEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """
    def __init__(self,
                 programs_dir: str = "programs/",
                 seed: int = 0,
                 max_turns: int = 10,
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

        self.dataset, self.eval_dataset = self.minihack_to_hf(train_list, eval_list)
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
            env_name = state["env_name"]

            env = NLELanguageWrapper(gym.make(env_name))
            obsv = env.reset()
            obsv = (
                f"\nText Glyphs:\n{obsv.get('text_glyphs', '').strip()}\n\n"
                f"Text Message:\n{obsv.get('text_message', '').strip()}\n\n"
                f"Text BLStats:\n{obsv.get('text_blstats', '').strip()}\n\n"
                f"Text Inventory:\n{obsv.get('text_inventory', '').strip()}\n\n"
                f"Text Cursor:\n{obsv.get('text_cursor', '').strip()}"
            )

            state['env'] = env

            available_actions = list(env.action_str_enum_map.keys()) #? not sure if this does what i want 

            state['available_actions'] = available_actions
            
            self.is_terminal = False
            state['initialized'] = True
            env_message = {"role": "user", "content": obsv}    
            return env_message, state
        
        # parse guess
        turn = self.parser.parse(messages[-1]["content"])
        action = str(turn.action)
        model_edit = turn.model_edit
        think = turn.think
        error_msg = None
        if action == "quit":
            self.is_terminal = True
        try: 
            obsv, reward, done, info = state['env'].step(action) # what to do with info and reward 
        except Exception as e:
            # execute NOP (wait) instead
            obsv, reward, done, info = state['env'].step("wait")
            error_msg = str(e)
        self.is_terminal = done 
        
        state['model'] = model_edit
        state['is_finished'] = self.is_terminal

        obsv = (
                f"\nText Glyphs:\n{obsv.get('text_glyphs', '').strip()}\n\n"
                f"Text Message:\n{obsv.get('text_message', '').strip()}\n\n"
                f"Text BLStats:\n{obsv.get('text_blstats', '').strip()}\n\n"
                f"Text Inventory:\n{obsv.get('text_inventory', '').strip()}\n\n"
                f"Text Cursor:\n{obsv.get('text_cursor', '').strip()}"
            )
        
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
            
        return completion, state

    def minihack_to_hf(self, train_list, eval_list) -> Tuple[Dataset, Dataset]:
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
        return dataset.repeat(100), eval_dataset.repeat(100)
    
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