import os
from openai import OpenAI

from verifiers.parsers import Parser
from verifiers.rubrics.rubric import Rubric

class TextWorldRubric(Rubric):
    def __init__(self, disable_rub: bool = False, **kwargs):
        super().__init__(**kwargs)
        if not disable_rub:
            self.add_reward_func(self.judge_reward_func)
    
    def judge_reward_func(self, state, **kwargs) -> float:
        # return cumulative reward from state? or can be averaged 
        return state['reward']
    
