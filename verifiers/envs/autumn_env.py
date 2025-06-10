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

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import ModelBasedRubric

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


DEFAULT_COLOR = "black"

# Define the color mapping for different grid entities
COLOR_MAP = {
    "gray": "gray",
    "gold": "gold",
    "green": "green",
    "mediumpurple": "mediumpurple",
    "purple": "purple",
    "white": "white",
    "yellow": "yellow",
    "blue": "blue",
    "red": "red",
    "orange": "orange",
    # Add more mappings as needed
}

# Automatically generate color to integer mapping for numerical processing
COLOR_TO_INT_MAP = {color: idx for idx, color in enumerate(COLOR_MAP.keys())}

def get_action_map(grid_size: int):
    acts = []
    acts = ["left", "right", "up", "down"]
    for i in range(grid_size):
        for j in range(grid_size):
            acts.append(f"click {i} {j}")
    acts.append("NOP")
    return {i: act for i, act in enumerate(acts)}


def get_environment_ids(base_dir: str):
    """
    Get list of environment names based on the `programs` folder, considering all the sexp files.
    """
    environment_ids = []
    for file in os.listdir(f"{base_dir}"):
        if file.endswith(".sexp"):
            environment_ids.append(file.split(".")[0])
    return environment_ids


def parse_grid(render_output: str):
    """
    Parses the JSON string output from render_all() into a grid and its size.

    Args:
        render_output (str): The JSON string representation of the grid.

    Returns:
        tuple: A tuple containing the grid dictionary and grid size.
    """
    try:
        elem_dict = json.loads(render_output)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {}, 0
    grid = elem_dict
    grid_size = elem_dict.pop("GRID_SIZE", 0)
    return grid, grid_size


def render_grid(grid: Dict[str, Any]) -> str:
    """
    Renders the grid into a string representation.

    Args:
        grid (Dict[str, Any]): The grid dictionary.
    
    """
    grid_size = grid.pop("GRID_SIZE", 0)
    grid_matrix = [[DEFAULT_COLOR for _ in range(grid_size)] for _ in range(grid_size)]
    for elem in grid:
        for subelem in grid[elem]:
            col_idx = subelem["position"]["x"]
            row_idx = subelem["position"]["y"]
            color_key = subelem["color"].lower()
            color = str(COLOR_TO_INT_MAP.get(color_key, color_key))
            if (row_idx >= 0 and row_idx < grid_size) and (col_idx >= 0 and col_idx < grid_size):
                grid_matrix[row_idx][col_idx] = color
    return '\n'.join([' '.join(row) for row in grid_matrix])

def render_grid_numbers(grid: Dict[str, Any]) -> str:
    """
    Renders the grid into a string representation.

    Args:
        grid (Dict[str, Any]): The grid dictionary.
    
    """
    grid_size = grid.pop("GRID_SIZE", 0)
    grid_matrix = [[DEFAULT_COLOR for _ in range(grid_size)] for _ in range(grid_size)]
    for elem in grid:
        for subelem in grid[elem]:
            col_idx = subelem["position"]["x"]
            row_idx = subelem["position"]["y"]
            color_key = subelem["color"].lower()
            color = COLOR_MAP.get(color_key, color_key)
            if (row_idx >= 0 and row_idx < grid_size) and (col_idx >= 0 and col_idx < grid_size):
                grid_matrix[row_idx][col_idx] = color
    return '\n'.join([' '.join(row) for row in grid_matrix])

SYSTEM_PROMPT = """You are a curious agent exploring an environment that consists of a grid containing cells which can take colors. 
You will be given observations and available actions to choose from at each step. 
Your task is to interact with the environment efficiently and effectively and try to understand the underlying rules of the environment.
Your understanding of the environment should be captured in the model text. 

The actions available are the following:
* left: press the left arrow key
* right: press the right arrow key
* up: press the up arrow key
* down: press the down arrow key
* click <x> <y>: click on the cell at location (<x>, <y>)
* NOP: do nothing
* quit: quit the environment

In each turn, think step-by-step inside <think>...</think> tags, generate the model in <model_edit>...</model_edit> tags, produce the action inside <action>...</action> tags.
Follow the format strictly."""


class AutumnEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """
    def __init__(self,
                 programs_dir: str = "programs/",
                 seed: int = 0,
                 max_turns: int = 10,
                 rubric: ModelBasedRubric = None,
                 std_lib_path: str = "autumn_stdlib.sexp",
                 train_list: List[str] = [],
                 eval_list: List[str] = [],
                 **kwargs):
        self.programs_dir = programs_dir
        self.seed = seed
        self.std_lib_path = std_lib_path
        self.render_mode = "json"
        parser = XMLParser(fields=["think", "model_edit" ,"action"], answer_field="action")
        self.dataset, self.eval_dataset = self.autumn_to_hf(train_list, eval_list)
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

    def render(self, render_dict: Dict[str, Any]):
        if self.render_mode == 'text':
            render_dict = render_grid(render_dict)
            return render_dict
        elif self.render_mode == 'numbers':
            render_dict = render_grid_numbers(render_dict)
            return render_dict
        elif self.render_mode == "json":
            return render_dict
        else:
            raise ValueError(f"Invalid mode: {self.render_mode}")

    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # load env 
        if 'initialized' not in state or not state['initialized']:
            env_name = state["env_name"]
            prog = open(f"{self.programs_dir}/{env_name}.sexp", "r").read()
            self.is_terminal = False
            self.interpreter = interpreter_module.Interpreter()
            stdlib = open(self.std_lib_path, "r").read()
            self.interpreter.run_script(prog, stdlib, "", self.seed)
            state['initialized'] = True
            observation = self.render(self.interpreter.render_all())
            env_message = {"role": "user", "content": observation}    
            return env_message, state
        
        # parse guess
        turn = self.parser.parse(messages[-1]["content"])
        action = str(turn.action)
        model_edit = turn.model_edit
        think = turn.think

        if action == "quit":
            self.is_terminal = True
        if action.startswith("click"):
            x, y = action.strip().split()[1:]
            self.interpreter.click(int(x), int(y))
        elif action == "left":
            self.interpreter.left()
        elif action == "right":
            self.interpreter.right()
        elif action == "up":
            self.interpreter.up()
        elif action == "down":
            self.interpreter.down()
        self.interpreter.step()
        observation = self.render(self.interpreter.render_all())
        
        # dmp = diff_match_patch()
        # dmp_patch = dmp.patch_fromText(model_edit)
        # state['model'] = dmp.patch_apply(dmp_patch, state['model'])
        state['model'] = model_edit
        state['is_finished'] = self.is_terminal

        env_message = {"role": "user", "content": observation}
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
        import pdb; pdb.set_trace();
        state = {'model': "", 'answer': answer, "env_name": prompt[-1]["content"]}
        messages = [{
            "role": "system",
            "content": self.system_prompt
        }]
        completion = []
        turn = 0
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

    def autumn_to_hf(self, train_list, eval_list) -> Tuple[Dataset, Dataset]:
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
        return dataset, eval_dataset