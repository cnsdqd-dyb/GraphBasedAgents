from typing import Dict, List, Tuple
from .abstract_agent import AbstractAgent
from langchain.agents import tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import random
import time
from langchain.load.dump import dumps
import json
from langchain.agents import tool
    

class Agent(AbstractAgent):
    base_url = "https://api.openai.com/v1"
    def __init__(self, name, env, local_port=5000, model="gpt-4-1106-preview", api_key_list=[], tools=[]):
        super().__init__(name, local_port, model, api_key_list)
        Agent.env = env
        self.tools = tools

    def medical_rescue_agent_tools():
        return [
            Agent.get_medical_resources,
            Agent.organize_medical_team,
            Agent.get_team_status,
            Agent.create_rescue_plan
        ]

    def emergency_rescue_agent_tools():
        return [
            Agent.get_rescue_resources,
            Agent.organize_rescue_team,
            Agent.get_team_status,
            Agent.deploy_rescue_team
        ]

    def security_control_agent_tools():
        return [
            Agent.get_security_resources,
            Agent.organize_security_team,
            Agent.get_team_status,
            Agent.deploy_security_team
        ]
    
    def disaster_monitoring_agent_tools():
        return [
            Agent.get_environmental_data,
            Agent.analyze_risk,
            Agent.get_team_status,
            Agent.predict_disaster_spread
        ]
    
    def traffic_control_agent_tools():
        return [
            Agent.get_traffic_status,
            Agent.plan_rescue_route,
            Agent.get_team_status,
            Agent.implement_traffic_control
        ]
    
    def tools():
        return [
            Agent.medical_rescue_agent_tools(),
            Agent.emergency_rescue_agent_tools(),
            Agent.security_control_agent_tools(),
            Agent.disaster_monitoring_agent_tools(),
            Agent.traffic_control_agent_tools()
        ]

    def run(self, instruction: str, player_name_list=[], max_turn=10):
        assert len(self.api_key_list) > 0, "Please set the api_key_list in Agent class."
        # dynamic api key
        action_list = []

        if "instruct" in self.model and "gpt" in self.model:
            from langchain.llms import OpenAI
            self.llm = OpenAI(model=self.model, temperature=0, max_tokens=256, openai_api_key=random.choice(self.api_key_list), base_url=Agent.base_url)
        elif "gpt" in self.model:
            from langchain.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(model=self.model, temperature=0, max_tokens=256, openai_api_key=random.choice(self.api_key_list), base_url=Agent.base_url)

        while max_turn > 0:
            try:
                agent = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    verbose=True,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    return_intermediate_steps=True,
                    max_execution_time=120,  # seconds
                    max_iterations=6,  # 决定了最大的迭代次数
                )
                
                input_text = f"You are in team {self.name}.\n{instruction}"
                if player_name_list:
                    input_text = f"You should control {player_name_list} work together.\n{instruction}"
                
                response = agent({"input":input_text})
                action_list = []
                response = json.loads(dumps(response, pretty=True))
                for step in response["intermediate_steps"]:
                    action_list.append({"action": step[0]["kwargs"], "feedback": step[1]})
                final_answer = response["output"]
                # save the action_list and final_answer

                with open(f"data/history/{hash(response['input'])}.json", "w") as f:
                    json.dump({"input": response["input"], "action_list": action_list, "final_answer": final_answer}, f,
                            indent=4)
                return final_answer, {"input": response["input"], "action_list": action_list, "final_answer": final_answer}

                        
            except Exception as e:
                print(f"Error occurred: {e}")
                print("Retrying...")
                time.sleep(1)
                max_turn -= 1
        
        return "Task execution failed.", {"input": instruction, "action_list": action_list, "final_answer": "Task execution failed."}
    
    # ----------------- Medical Rescue Agent -----------------
    @tool
    def get_medical_resources() -> Dict:
        """获取可用的医疗资源信息"""
        resources = []
        for resource_id, resource in Agent.env.resources.items():
            if (resource.type == "vehicle" and resource.properties.get("vehicle_type") == "ambulance") or \
               (resource.type == "personnel" and resource.properties.get("role") == "medic") or \
               (resource.type == "equipment" and resource.properties.get("equipment_type") == "medical"):
                if resource.status == "available":
                    resources.append(resource.to_dict())
        return {
            "message": {"available_resources": resources},
            "status": True
        }
        
    @tool
    def organize_medical_team(unit_name: str, required_resources: Dict[str, int]) -> Dict:
        """
        Args:
            required_resources: 需要的资源数量，例如 ("ambulance": 1, "medic": 2, "medical_equipment": 2)
        """
        assigned_resources = []
        
        # 遍历所有资源类型
        for resource_type, count in required_resources.items():
            assigned_count = 0
            for resource_id, resource in Agent.env.resources.items():
                if assigned_count >= count:
                    break
                    
                # 检查资源类型是否匹配
                if (resource_type == "ambulance" and resource.type == "vehicle" and 
                    resource.properties.get("vehicle_type") == "ambulance") or \
                   (resource_type == "medic" and resource.type == "personnel" and 
                    resource.properties.get("role") == "medic") or \
                   (resource_type == "medical_equipment" and resource.type == "equipment" and 
                    resource.properties.get("equipment_type") == "medical"):
                    
                    if resource.status == "available":
                        if Agent.env.assign_resource(resource_id, unit_name):
                            assigned_resources.append(resource.to_dict())
                            assigned_count += 1
                            
        if len(assigned_resources) < sum(required_resources.values()):
            return {
                "message": {"assigned_resources": assigned_resources},
                "status": False
            }
            
        return {
            "message": {"assigned_resources": assigned_resources},
            "status": True
        }

    @tool
    def get_team_status(unit_name: str) -> Dict:
        """获取当前医疗队伍的状态"""
        status = Agent.env.get_agent_resources(unit_name)
        return {
            "message": {"team_resources": status},
            "status": True
        }
        
    @tool
    def create_rescue_plan(unit_name: str, event_location: Tuple[float, float]) -> Dict:
        """基于事件位置创建救援计划"""
        team_status = Agent.get_team_status(unit_name)
        if not team_status["message"]["team_resources"]:
            return {
                "message": "No team resources available for rescue",
                "status": False
            }
            
        # 更新所有团队资源的位置
        for resource in team_status["message"]["team_resources"]:
            Agent.env.update_resource_status(
                resource["id"],
                resource["status"],
                event_location
            )
            
        return {
            "message": {
                "event_location": event_location,
                "team_resources": team_status["message"]["team_resources"]
            },
            "status": True
        }
    
    # ----------------- Emergency Rescue Agent -----------------
    @tool
    def get_rescue_resources() -> Dict:
        """获取可用的救援资源信息"""
        resources = []
        for resource_id, resource in Agent.env.resources.items():
            if (resource.type == "vehicle" and resource.properties.get("vehicle_type") == "fire_truck") or \
               (resource.type == "personnel" and resource.properties.get("role") == "firefighter"):
                if resource.status == "available":
                    resources.append(resource.to_dict())
        return {
            "message": {"available_resources": resources},
            "status": True
        }
        
    @tool
    def organize_rescue_team(unit_name: str, required_resources: Dict[str, int]) -> Dict:
        """
        组织救援队伍
        Args:
            unit_name: emergency_rescue_agent / traffic_control_agent / disaster_monitoring_agent / security_control_agent / medical_rescue_agent
            required_resources: 需要的资源数量，例如 ("fire_truck": 1, "firefighter": 4)
        """
        assigned_resources = []
        
        for resource_type, count in required_resources.items():
            assigned_count = 0
            for resource_id, resource in Agent.env.resources.items():
                if assigned_count >= count:
                    break
                    
                if (resource_type == "fire_truck" and resource.type == "vehicle" and 
                    resource.properties.get("vehicle_type") == "fire_truck") or \
                   (resource_type == "firefighter" and resource.type == "personnel" and 
                    resource.properties.get("role") == "firefighter"):
                    
                    if resource.status == "available":
                        if Agent.env.assign_resource(resource_id, unit_name):
                            assigned_resources.append(resource.to_dict())
                            assigned_count += 1
                            
        if len(assigned_resources) < sum(required_resources.values()):
            return {
                "message": {"assigned_resources": assigned_resources},
                "status": False
            }
            
        return {
            "message": {"assigned_resources": assigned_resources},
            "status": True
        }

    @tool
    def deploy_rescue_team(unit_name: str, location: Tuple[float, float]) -> Dict:
        """部署救援队伍到指定位置"""
        team_status = Agent.get_team_status(unit_name)
        if not team_status["message"]["team_resources"]:
            return {
                "message": "No team resources available for deployment",
                "status": False
            }
            
        # 更新所有团队资源的位置
        for resource in team_status["message"]["team_resources"]:
            Agent.env.update_resource_status(
                resource["id"],
                resource["status"],
                location
            )
            
        return {
            "message": {
                "deployment_location": location,
                "team_resources": team_status["message"]["team_resources"]
            },
            "status": True
        }
    
    # ----------------- Security Control Agent -----------------
    @tool
    def get_security_resources() -> Dict:
        """获取可用的安保资源信息"""
        resources = []
        for resource_id, resource in Agent.env.resources.items():
            if (resource.type == "vehicle" and resource.properties.get("vehicle_type") == "police_car") or \
               (resource.type == "personnel" and resource.properties.get("role") == "police"):
                if resource.status == "available":
                    resources.append(resource.to_dict())
        return {
            "message": {"available_resources": resources},
            "status": True
        }
        
    @tool
    def organize_security_team(unit_name: str, required_resources: Dict[str, int]) -> Dict:
        """
        组织安保队伍
        Args:
            unit_name: emergency_rescue_agent / traffic_control_agent / disaster_monitoring_agent / security_control_agent / medical_rescue_agent
            required_resources: 需要的资源数量，例如 ("police_car": 2, "police": 4)
        """
        assigned_resources = []
        
        for resource_type, count in required_resources.items():
            assigned_count = 0
            for resource_id, resource in Agent.env.resources.items():
                if assigned_count >= count:
                    break
                    
                if (resource_type == "police_car" and resource.type == "vehicle" and 
                    resource.properties.get("vehicle_type") == "police_car") or \
                   (resource_type == "police" and resource.type == "personnel" and 
                    resource.properties.get("role") == "police"):
                    
                    if resource.status == "available":
                        if Agent.env.assign_resource(resource_id, unit_name):
                            assigned_resources.append(resource.to_dict())
                            assigned_count += 1
                            
        if len(assigned_resources) < sum(required_resources.values()):
            return {
                "message": {"assigned_resources": assigned_resources},
                "status": False
            }
            
        return {
            "message": {"assigned_resources": assigned_resources},
            "status": True
        }

    @tool
    def deploy_security_team(unit_name:str, locations: List[Tuple[float, float]]) -> Dict:
        """在多个位置部署安保队伍"""
        team_status = Agent.get_team_status(unit_name)
        if not team_status["message"]["team_resources"]:
            return {
                "message": "No team resources available for deployment",
                "status": False
            }
            
        # 将资源平均分配到各个位置
        resources = team_status["message"]["team_resources"]
        resources_per_location = len(resources) // len(locations)
        
        deployments = []
        for i, location in enumerate(locations):
            start_idx = i * resources_per_location
            end_idx = start_idx + resources_per_location if i < len(locations) - 1 else len(resources)
            
            location_resources = resources[start_idx:end_idx]
            for resource in location_resources:
                Agent.env.update_resource_status(
                    resource["id"],
                    resource["status"],
                    location
                )
                deployments.append({
                    "resource": resource,
                    "location": location
                })
                
        return {
            "message": {
                "deployment_locations": locations,
                "deployments": deployments
            },
            "status": True
        }

    # ----------------- Disaster Monitoring Agent -----------------
    @tool
    def get_environmental_data() -> Dict:
        """获取环境数据"""
        success, data, msg = Agent.env.get_environmental_data()
        if success:
            return {"message": data, "status": True}
        else:
            return {"message": msg, "status": False}

    @tool
    def analyze_risk(env_data: Dict) -> Dict:
        """分析风险"""
        if "gas_concentration" not in env_data:
            return {"message": "Invalid environmental data", "status": False}
            
        risk_level = "HIGH" if env_data["gas_concentration"] > 10.0 else \
                    "MEDIUM" if env_data["gas_concentration"] > 5.0 else "LOW"
                    
        return {
            "message": {
                "level": risk_level,
                "factors": {
                    "gas_concentration": env_data["gas_concentration"],
                    "wind_speed": env_data["wind_speed"]
                }
            },
            "status": True
        }

    @tool
    def predict_disaster_spread(current_data: Dict) -> Dict:
        """预测灾害蔓延"""
        if "wind_direction" not in current_data or "wind_speed" not in current_data:
            return {"message": "Missing wind data", "status": False}
            
        # Simple prediction based on wind
        spread_direction = current_data["wind_direction"]
        spread_speed = current_data["wind_speed"] * 0.5
        
        return {
            "message": {
                "direction": spread_direction,
                "speed": spread_speed,
                "affected_area_growth": f"{spread_speed * 10:.1f} meters per minute"
            },
            "status": True
        }

    # ----------------- Traffic Control Agent -----------------
    @tool
    def get_traffic_status(self) -> Dict:
        """获取当前交通状态"""
        status, roads, desc = Agent.env.get_traffic_info()
        if not status:
            return {"message": desc, "status": False}
        
        return {"message": roads, "status": True}

    @tool
    def plan_rescue_route(start: tuple, end: tuple) -> Dict:
        """规划救援路线"""
        status, roads, desc = Agent.env.get_traffic_info()
        if not status:
            return {"message": desc, "status": False}
            
        # Simple direct route for now
        return {
            "message": {
                "start": start,
                "end": end,
                "distance": ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5,
                "estimated_time": "10 minutes"
            },
            "status": True
        }

    @tool
    def implement_traffic_control(road_ids: List[str]) -> Dict:
        """实施交通管制"""
        status, roads, desc = Agent.env.get_traffic_info()
        if not status:
            return {"message": desc, "status": False}
            
        controlled_roads = [r for r in road_ids if r in roads]
        
        return {
            "message": {
                "controlled_roads": controlled_roads,
                "status": "implemented",
                "affected_areas": len(controlled_roads)
            },
            "status": True
        }