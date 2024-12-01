# Standard library imports
import sys
import os
import time
import logging
from typing import Dict, List, Tuple
from random import random, randint, choice

# Add project root to path
sys.path.append(os.getcwd())

# Project imports
from CityPipe.utils import *
from type_define.graph import Task
from CityEnvironment.city_emergency_env import CityEmergencyEnv
from CityPipe.data_manager import DataManager
from LLM.openai_models import OpenAILanguageModel
from CityPipe.agent_prompt import (
    city_emergency_knowledge,
    agent_prompt,
    agent_cooper_prompt,
    reflect_system_prompt,
    reflect_user_prompt
)

class AgentFeedback:
    def __init__(self, task:Task, detail, status):
        self.task = task
        self.detail = detail
        self.status = status

    def to_json(self) -> dict:
        return {
            "task": self.task.to_json(),
            "detail": self.detail,
            "status": self.status,
        }

class BaseAgent:
    '''
    ### BaseAgent is the single agent in the system, it can take action and reflect
    
    step: take an action and return the feedback and detail
    reflect: reflect on the task and return the result
    to_json: return the json format of the agent
    get_status: get the current status of the agent and its resources
    '''
    
    _virtual_debug = False
    
    def __init__(self, llm:OpenAILanguageModel , env:CityEmergencyEnv, data_manager:DataManager, 
                 name:str, logger:logging.Logger = None, silent = False, **kwargs):
        self.env = env
        self.name = name
        self.data_manager = data_manager
        self.llm = llm
        self.history_action_list = ["No action yet"]
        self.agent_type = "general" if "agent_type" not in kwargs else kwargs["agent_type"]
        self.logger = logger
        if not env.running:
            BaseAgent._virtual_debug = True
        
        if self.logger is None:
            self.logger = init_logger("BaseAgent", dump=True, silent=silent)
            
    def get_status(self) -> Dict:
        """获取agent当前状态，包括分配给它的所有资源状态"""
        status = {
            "name": self.name,
            "type": self.agent_type,
            "last_action": self.history_action_list[-1] if self.history_action_list else None,
            "resources": self.env.get_agent_resources(self.name)
        }
        status = {"message": str(status), "status": True}
        return status

    def step(self, task:Task) -> (str, dict):
        '''
        take an action and return the feedback and detail
        return: final_answer, {"input": response["input"], "action_list": action_list, "final_answer": final_answer}
        '''
        if BaseAgent._virtual_debug:
            return self.virtual_step(task)
        if len(task._agent) == 1:
            task_str = format_string(agent_prompt, {
                "task_description": task.description, 
                "milestone_description": task.milestones, 
                "env": self.data_manager.query_env_with_task(task.description),
                "relevant_data": smart_truncate(task.content, max_length=4096),
                "agent_name": self.name,
                "agent_type": self.agent_type,
                "agent_state": self.data_manager.query_history(self.name),
                "other_agents": self.other_agents(),
                "agent_action_list": self.history_action_list,
                "city_emergency_knowledge": city_emergency_knowledge,
            })
        else:
            task_str = format_string(agent_cooper_prompt, {
                "task_description": task.description, 
                "milestone_description": task.milestones, 
                "env": self.data_manager.query_env_with_task(task.description),
                "relevant_data": smart_truncate(task.content, max_length=4096),
                "agent_name": self.name,
                "agent_state": self.data_manager.query_history(self.name),
                "other_agents": self.other_agents(),
                "agent_action_list": self.history_action_list,
                "team_members": ", ".join(task._agent),
                "city_emergency_knowledge": city_emergency_knowledge,
            })
            
        self.logger.debug("="*20 + " Agent Step " + "="*20)
        self.logger.info(f"{self.name} try task:\n {task.description}")
        self.logger.info(f"{self.history_action_list}")
        self.logger.info(f"other agents: {self.other_agents()}")
        self.logger.info(f"{self.name} status:\n {self.data_manager.query_history(self.name)}")
        max_retry = 3
        while max_retry > 0:
            try:
                feedback, detail = self.env.step(self.name, task_str)
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
                max_retry -= 1
                time.sleep(3)
        status = self.get_status()
        self.data_manager.update_database(AgentFeedback(task, detail, status).to_json())
        # self.data_manager.save()
        return feedback, detail
    
    def other_agents(self) -> [str]:
        '''
        return the feedback of other agent's pretask
        '''
        return self.data_manager.query_other_agent_state(self.name)
    
    def action_format(self, action:dict) -> str:
        action_str = '''{{message}}'''
        print(action["feedback"])
        return format_string(action_str, action["feedback"])
    
    def reflect(self, task: Task, detail) -> bool:
        '''
        Reflect on the task and return the result
        '''
        task_description = task.description
        milestone_description = task.milestones
        action_history = detail["action_list"]
        global reflect_system_prompt, reflect_user_prompt
        if isinstance(self.llm, OpenAILanguageModel):
            prompt = format_string(reflect_user_prompt,
                                   {
                                       "task_description": task_description,
                                       "milestone_description": milestone_description,
                                       "state": self.data_manager.query_history(self.name),
                                       "action_history": action_history
                                   })
            response = self.llm.generate(reflect_system_prompt, prompt, cache_enabled=False, max_tokens=256, json_check=True)
        else:
            prompt = format_string(reflect_user_prompt,
                                   {
                                       "task_description": task_description,
                                       "milestone_description": milestone_description,
                                       "state": self.data_manager.query_history(self.name),
                                       "action_history": action_history
                                   })
            response = self.llm.generate(reflect_system_prompt, prompt, cache_enabled=False, max_tokens=256, json_check=True)
        # print(response)
        result = extract_info(response)[0]
        task.reflect = result
        task._summary.append(result["summary"])

        # add the action to the history
        self.history_action_list = [self.action_format(action) for action in action_history]
        return result["task_status"]
    
    def to_json(self) -> dict:
        return {
            "name": self.name
        }