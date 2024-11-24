from abc import ABC, abstractmethod
import json
import os
import time
import subprocess
from functools import wraps
import requests
from langchain.agents import tool, initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.load.dump import dumps

class AbstractAgent(ABC):
    """
    AbstractAgent is a base class for any agent. It defines common interfaces
    and functionalities that all agents must implement.
    """

    headers = {'Content-Type': 'application/json'}
    agent_processes = {}

    def __init__(self, name, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        self.name = name
        self.local_port = local_port
        self.model = model
        self.api_key_list = api_key_list

    @abstractmethod
    def run(self, instruction: str, **kwargs):
        """
        Run the agent with a given instruction.
        """
        pass

    @staticmethod
    def timeit(func):
        """
        Decorator to measure the execution time of a function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function {func.__name__} took {end_time - start_time} seconds")
            return result
        return wrapper

    def save_action_log(self, agent_name, func_name, start_time, end_time, duration, kwargs, result):
        """
        Save the action log to a file.
        """
        if not os.path.exists("data"):
            os.makedirs("data")

        max_try = 3
        while max_try:
            try:
                action_log_path = "data/action_log.json"
                if os.path.exists(action_log_path):
                    with open(action_log_path, "r") as f:
                        action_log = json.load(f)
                else:
                    action_log = {}
                if agent_name not in action_log:
                    action_log[agent_name] = []

                action_log[agent_name].append({
                    "action": func_name,
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                    "duration": duration,
                    "kwargs": kwargs,
                    "result": result,
                })

                with open(action_log_path, "w") as f:
                    json.dump(action_log, f, indent=4)
                break
            except Exception as e:
                print(e)
                max_try -= 1
                time.sleep(1)

    @staticmethod
    def launch_subprocess(name, script, host="localhost", port=25565, local_port=1024,world="world", verbose=False, ignore_names=[], debug=False, fast=False):
        """
        Launch the agent subprocess.
        """
        if verbose:
            print(f"Launching agent {name}...")

        try:
            AbstractAgent.agent_processes[name] = subprocess.Popen(
                ["python", script, "-H", host, "-P", str(port), "-LP", str(local_port), "-U", name, "-W", world, "-D", str(debug)],
                shell=False
            )
            time.sleep(10 if fast else 3)
        except Exception as e:
            print(f"An error occurred while launching agent {name}: {e}")
            time.sleep(1)

        if verbose:
            print(f"Agent {name} launched.")

    @staticmethod
    def launch(host="localhost", port=25565, world="world", verbose=False, ignore_names=[], debug=False, fast=False):
        """
        Launch the environment for the agents.
        """
        if verbose:
            print("Launching agents...")

        for agent_name, local_port in AbstractAgent.agent_processes.items():
            if agent_name in ignore_names:
                continue
            
            AbstractAgent.launch_subprocess(agent_name, "agent.py", host, port, local_port, world, verbose, ignore_names, debug, fast)

        if verbose:
            print("Agents launched.")

    @staticmethod
    def kill():
        """
        Terminate all running agent processes.
        """
        for process in AbstractAgent.agent_processes.values():
            process.terminate()

class MinecraftAgent(AbstractAgent):
    def __init__(self, name, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        AbstractAgent.agent_processes[name] = local_port
        self.tools = []  # Define your tools here

    @tool
    @AbstractAgent.timeit
    def example_tool(self, player_name: str):
        """Example tool using LangChain."""
        url = f"http://localhost:{self.local_port}/example"
        data = {"player_name": player_name}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def run(self, instruction: str, player_name_list=[], max_turn=10):
        assert len(self.api_key_list) > 0, "Please set the api_key_list."
        # Initialize the language model
        from langchain.llms import OpenAI
        llm = OpenAI(model=self.model, temperature=0, max_tokens=256, openai_api_key=self.api_key_list[0])

        while max_turn > 0:
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                verbose=True,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                return_intermediate_steps=True,
                max_execution_time=120,
                max_iterations=6,
            )
            agent.handle_parsing_errors = True
            response = None
            try:
                with get_openai_callback() as cb:
                    start_time = time.time()
                    if len(player_name_list) == 0:
                        response = agent({"input": f"Your name is {self.name}.\n{instruction}"})
                    else:
                        response = agent({"input": f"You should control {player_name_list} work together.\n{instruction}"})
                    end_time = time.time()
                break
            except Exception as e:
                print(e)
                print("Retrying...")
                time.sleep(1)
                max_turn -= 1

        if max_turn == 0 or response is None:
            return "The task execution failed.", {}

        action_list = []
        response_data = json.loads(dumps(response, pretty=True))
        for step in response_data["intermediate_steps"]:
            action_list.append({"action": step[0]["kwargs"], "feedback": step[1]})
        final_answer = response_data["output"]

        with open(f"data/history/{hash(response_data['input'])}.json", "w") as f:
            json.dump({"input": response_data["input"], "action_list": action_list, "final_answer": final_answer}, f, indent=4)

        return final_answer, {"input": response_data["input"], "action_list": action_list, "final_answer": final_answer}

if __name__ == "__main__":
    agent = MinecraftAgent(name="Steve", local_port=5004, api_key_list=["your_api_key_here"])
    AbstractAgent.launch(host="10.21.31.18", port=25565, verbose=True)
    prompt = "try to find a chest and open it, do not stop until you find it."
    agent.run(prompt)
