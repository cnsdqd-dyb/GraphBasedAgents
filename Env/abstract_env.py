from abc import ABC, abstractmethod
from ..Agent.abstract_agent import AbstractAgent
import logging
import json
import os
import time

class MultiAgentEnvironment(ABC):
    def __init__(self, env_type, task_id, host="0.0.0.0", port=25565, task_name="test", virtual_debug=False):
        self.env_type = env_type
        self.task_id = task_id
        self.host = host
        self.port = port
        self.task_name = task_name
        self.agent_pool = []
        self.log = {}
        self.running = False
        self.virtual_debug = virtual_debug
        self.logger = self.init_logger(name="Env", level=logging.DEBUG)
        self.launch_time = None

        # Ensure necessary directories exist
        self.prepare_directories()
        self.reset_token()

    def prepare_directories(self):
        if not os.path.exists("data"):
            os.mkdir("data")

        if not os.path.exists("data/history"):
            os.mkdir("data/history")

        with open("data/score.json", "w") as f:
            json.dump({}, f, indent=4)

        with open("data/action_log.json", "w") as f:
            json.dump({}, f, indent=4)

        with open("data/llm_inference.json", "w") as f:
            json.dump({"time": 0}, f, indent=4)

        with open(".cache/state.json", "w") as f:
            json.dump({"state": "idle"}, f)

    def init_logger(self, name, level):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    @abstractmethod
    def agent_register(self, agent_tool=[], agent_number: int = 1, name_list: [str] = []):
        '''
        register the agent to the environment
        '''
        if len(name_list) != agent_number:
            self.logger.warning(
                "[warning but dont worry] agent number not equal to names number, random names will be used")


    @abstractmethod
    def launch(self, debug: bool = False, fast_api=False):
        AbstractAgent.launch(host=self.host, port=self.port, world=self.task_name, verbose=True, debug=debug, fast=fast_api)
        self.running = True
        self.reset()

    @abstractmethod
    def step(self, agent_name: str, action: str, max_turn: int = 2):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def reset_token(self):
        tokens = {
            "dates": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
            "total_cost": 0,
            "action_cost": 0
        }
        with open("data/tokens.json", "w") as f:
            json.dump(tokens, f, indent=4)

    def get_total_time(self):
        if self.launch_time is None:
            return 0
        return time.time() - self.launch_time

    def get_action_log(self):
        if os.path.exists("data/action_log.json"):
            with open("data/action_log.json") as f:
                action_log = json.load(f)
            return action_log
        else:
            return {"message": "action log not found", "status": False}

# Example usage:
# class VillagerBench(MultiAgentEnvironment):
#     def agent_register(self, agent_tool=[], agent_number: int = 1, name_list: [str] = []):
#         ...
#     def launch(self, debug: bool = False, fast_api=False):
#         ...
#     def step(self, agent_name: str, action: str, max_turn: int = 2):
#         ...
#     def stop(self):
#         ...

