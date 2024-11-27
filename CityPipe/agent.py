import sys
import os
sys.path.append(os.getcwd())
import time
import logging
from CityEnvironment.city_emergency_env import CityEmergencyEnv
from type_define.graph import Task
from CityPipe.data_manager import DataManager
from CityPipe.utils import *
from LLM.openai_models import OpenAILanguageModel
from random import random, randint, choice
from CityPipe.agent_prompt import *

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

class EmergencyResponseAgent:
    '''
    ### EmergencyResponseAgent is the base agent class for emergency response scenarios
    
    Capabilities:
    - Respond to emergency situations
    - Coordinate with other emergency response units
    - Manage and allocate resources
    - Monitor and update situation status
    
    Methods:
    - step: take an action and return the feedback
    - reflect: evaluate response effectiveness and plan next steps
    - coordinate: communicate with other response units
    - to_json: serialize agent state
    '''
    _virtual_debug = False
    def __init__(self, llm:OpenAILanguageModel, env:CityEmergencyEnv, data_manager:DataManager, 
                 name:str, agent_type:str, logger:logging.Logger = None, silent = False, **kwargs):
        self.env = env
        self.name = name
        self.agent_type = agent_type  # e.g. medical, fire, police
        self.data_manager = data_manager
        self.llm = llm
        self.history_action_list = ["No action yet"]
        self.resources = {}  # Available resources
        self.location = None  # Current location
        self.status = "available"  # Agent status

        self.logger = logger
        if not env.running:
            EmergencyResponseAgent._virtual_debug = True

        if self.logger is None:
            self.logger = init_logger("EmergencyResponseAgent", dump=True, silent=silent)

    def step(self, task:Task) -> (str, dict):
        '''
        Take an emergency response action and return the feedback
        Returns: feedback, {"input": response["input"], "action_list": action_list, "final_answer": final_answer}
        '''
        if EmergencyResponseAgent._virtual_debug:
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
                "agent_resources": self.resources,
                "agent_location": self.location,
                "emergency_knowledge_base": emergency_knowledge_base})
        else:
            task_str = format_string(agent_cooper_prompt, {
                "task_description": task.description, 
                "milestone_description": task.milestones,
                "env": self.data_manager.query_env_with_task(task.description),
                "relevant_data": smart_truncate(task.content, max_length=4096),
                "agent_name": self.name,
                "agent_type": self.agent_type, 
                "agent_state": self.data_manager.query_history(self.name),
                "other_agents": self.other_agents(),
                "agent_action_list": self.history_action_list,
                "team_members": ", ".join(task._agent),
                "agent_resources": self.resources,
                "agent_location": self.location,
                "emergency_knowledge_base": emergency_knowledge_base})

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
        status = self.env.agent_status(self.name)
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
    
    def virtual_env(name:str):
        '''
        Virtual environment for testing emergency response scenarios
        '''
        env = {
            "resources": {
                "ambulances": 2,
                "fire_trucks": 1,
                "police_units": 3,
                "medical_supplies": 100,
                "water_resources": 1000
            },
            "incidents": [
                {
                    "type": "fire",
                    "location": [40.7128, -74.0060],
                    "severity": "high",
                    "casualties": 0,
                    "status": "active"
                }
            ],
            "agent_type": "medical",
            "agent_name": name,
            "agent_location": [40.7130, -74.0065],
            "agent_status": "available",
            "weather": {
                "condition": "clear",
                "temperature": 25,
                "wind_speed": 10
            }
        }
        return env

    def virtual_step(self, task:Task) -> (str, dict):
        '''
        Virtual step for testing emergency response scenarios
        '''
        # random action
        action = choice(["respond", "coordinate", "allocate_resources", "monitor_situation"])
        input = smart_truncate(task.to_json(), max_length=4096)
        random_action_num = randint(1, 10)
        action_list = []
        for i in range(random_action_num):
            action_dict = {
                "tool" : action,
                "tool_input" : {
                    "player_name": self.name,
                    "x": randint(-100, 100),
                    "y": randint(-100, 100),
                    "z": randint(-100, 100),
                },
                "log": "random action"
            }
            feedback = {
                "message": f"execute {action_dict['tool']} at {action_dict['tool_input']['x']} {action_dict['tool_input']['y']} {action_dict['tool_input']['z']}",
                "status": True
            }
            action_list.append({"action": action_dict, "feedback": feedback})
        score = random()
        if score > 0.3:
            final_answer = f"successfully responded to {task.description}."
            task.status = Task.success
        else:
            final_answer = f"failed to respond to {task.description}."
            task.status = Task.failure
        detail = {
            "input": input,
            "action_list": action_list,
            "final_answer": final_answer,
        }
        
        self.data_manager.update_database(AgentFeedback(task, detail, CityEmergencyEnv.virtual_env(self.name)).to_json())
        # self.data_manager.save()
        return final_answer, {"input": input, "action_list": action_list, "final_answer": final_answer}

class BaseAgent:
    '''
    ### BaseAgent is the single agent in the system, it can take action and reflect
    
    step: take an action and return the feedback and detail
    reflect: reflect on the task and return the result
    to_json: return the json format of the agent
    '''
    _virtual_debug = False
    def __init__(self, llm:OpenAILanguageModel , env:CityEmergencyEnv, data_manager:DataManager, name:str, logger:logging.Logger = None, silent = False, **kwargs):
        self.env = env
        self.name = name
        self.data_manager = data_manager
        self.llm = llm
        self.history_action_list = ["No action yet"]

        self.logger = logger
        if not env.running:
            BaseAgent._virtual_debug = True

        if self.logger is None:
            self.logger = init_logger("BaseAgent", dump=True, silent=silent)

    def step(self, task:Task) -> (str, dict):
        '''
        take an action and return the feedback and detail
        return: final_answer, {"input": response["input"], "action_list": action_list, "final_answer": final_answer}
        '''
        if BaseAgent._virtual_debug:
            return self.virtual_step(task)
        if len(task._agent) == 1:
            task_str = format_string(agent_prompt, {"task_description": task.description, "milestone_description": task.milestones, 
                                    "env": self.data_manager.query_env_with_task(task.description),
                                    "relevant_data": smart_truncate(task.content, max_length=4096), # TODO: change to "relevant_data": task.content
                                    "agent_name": self.name,
                                    "agent_state": self.data_manager.query_history(self.name),
                                    "other_agents": self.other_agents(),
                                    "agent_action_list": self.history_action_list,
                                    "minecraft_knowledge_card": minecraft_knowledge_card})
        else:
            task_str = format_string(agent_cooper_prompt, {"task_description": task.description, "milestone_description": task.milestones, 
                                    "env": self.data_manager.query_env_with_task(task.description),
                                    "relevant_data": smart_truncate(task.content, max_length=4096), # TODO: change to "relevant_data": task.content
                                    "agent_name": self.name,
                                    "agent_state": self.data_manager.query_history(self.name),
                                    "other_agents": self.other_agents(),
                                    "agent_action_list": self.history_action_list,
                                    "team_members": ", ".join(task._agent),
                                    "minecraft_knowledge_card": minecraft_knowledge_card})
            
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
        status = self.env.agent_status(self.name)
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
    