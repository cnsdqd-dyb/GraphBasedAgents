import datetime
import numpy as np
from langchain.retrievers.multi_vector import SearchType
from typing import Dict, List, Tuple, Optional

from LLM.openai_models import OpenAILanguageModel
from CityPipe.data_prompt import AGENT_INFO_STORE_FORMAT, \
    SUCCESS_DECOMPOSE_PLAN_FORMAT, NOT_SUCCESS_DECOMPOSE_PLAN_FORMAT, ENVIRONMENT_UPDATE_FORMAT, \
    HISTORY_SUMMARY_PROMPT, \
    SUMMARY_ENVIRONMENT_SYSTEM_PROMPT, SUMMARY_ENVIRONMENT_EXAMPLE_PROMPT
from type_define.decomposed_summary_system import DecomposeSummarySystem
from type_define.task_summary_tree import TaskSummaryTree
from type_define.graph import Task, Graph
from CityPipe.utils import *
from CityEnvironment.city_emergency_env import CityEmergencyEnv, EmergencyEvent
from CityEnvironment.city_map import Building
import logging
import random

class DataManager:
    '''
    DataManager is responsible for managing the data of the emergency environment, agents, history, and experience.
    Key responsibilities:
    1. Track emergency events and their status
    2. Monitor resource allocation and availability
    3. Maintain agent histories and states
    4. Provide environment information queries
    '''
    def __init__(self, llm:OpenAILanguageModel=None, logger:logging.Logger=None):
        self._env_data = {
            "events": {},      # 活动事件
            "buildings": {},   # 建筑信息
            "departments": {   # 部门信息
                "fire_department": {
                    "locations": [],        # 消防站位置列表
                    "personnel": {          # 人员信息
                        "firefighters": [],  # 消防员列表
                        "commanders": [],    # 指挥官列表
                    },
                    "functions": ["fire_control", "rescue", "hazmat_handling"],  # 部门职能
                    "on_duty": {},         # 当前在岗人员
                    "contacts": {          # 联系方式
                        "emergency": "",    # 紧急联系电话
                        "office": "",       # 办公电话
                        "email": ""         # 电子邮件
                    }
                },
                "police_department": {
                    "locations": [],        # 警察局位置列表
                    "personnel": {
                        "officers": [],     # 警察列表
                        "commanders": [],   # 指挥官列表
                    },
                    "functions": ["law_enforcement", "traffic_control", "evacuation"],
                    "on_duty": {},
                    "contacts": {
                        "emergency": "",
                        "office": "",
                        "email": ""
                    }
                },
                "medical_department": {
                    "locations": [],        # 医院和急救中心位置列表
                    "personnel": {
                        "doctors": [],      # 医生列表
                        "nurses": [],       # 护士列表
                        "paramedics": []    # 急救人员列表
                    },
                    "functions": ["emergency_medical_care", "patient_transport"],
                    "on_duty": {},
                    "contacts": {
                        "emergency": "",
                        "office": "",
                        "email": ""
                    }
                }
            },
            "resources": {     # 资源状态
                "ambulances": [],
                "fire_trucks": [],
                "police_cars": []
            },
            "traffic": None,   # 交通状态
        }
        self._history_data = {}  # 历史记录
        self._experience_data = {}  # 经验数据
        self._agent_data = []  # 智能体数据
        self.llm = llm
        self._logger = logger
        if self._logger is None:
            self._logger = init_logger("DataManager", dump=True)

    def update_environment(self, env: CityEmergencyEnv):
        """更新环境数据"""
        # 更新事件信息
        self._env_data["events"] = {}
        for event_id, event in env.active_events.items():
            self._env_data["events"][event_id] = {
                "type": event.type,
                "location": event.location,
                "floor": event.floor,
                "severity": event.severity,
                "casualties": event.casualties,
                "affected_radius": event.affected_radius,
                "properties": event.properties,
                "is_active": event.is_active
            }
        
        # 更新建筑信息
        self._env_data["buildings"] = {}
        for building_id, building in env.city_map.buildings.items():
            self._env_data["buildings"][building_id] = {
                "type": building.type,
                "location": building.location,
                "floors": building.floors,
                "capacity": building.capacity,
                "resources": building.resources
            }
            
            # 根据建筑类型更新部门位置信息
            if building.type == "fire_station":
                self._env_data["departments"]["fire_department"]["locations"].append({
                    "id": building_id,
                    "location": building.location
                })
            elif building.type == "police_station":
                self._env_data["departments"]["police_department"]["locations"].append({
                    "id": building_id,
                    "location": building.location
                })
            elif building.type == "hospital":
                self._env_data["departments"]["medical_department"]["locations"].append({
                    "id": building_id,
                    "location": building.location
                })
        
        # 更新部门人员和在岗信息
        for dept_name, dept_data in env.departments.items():
            if dept_name in self._env_data["departments"]:
                dept = self._env_data["departments"][dept_name]
                # 更新人员信息
                for role, personnel in dept_data.personnel.items():
                    if role in dept["personnel"]:
                        dept["personnel"][role] = personnel
                # 更新在岗信息
                dept["on_duty"] = dept_data.on_duty
                # 更新联系方式
                if hasattr(dept_data, "contacts"):
                    dept["contacts"] = dept_data.contacts
        
        # 更新资源状态
        self._env_data["resources"] = env.deployed_resources
        
        # 更新交通状态
        self._env_data["traffic"] = env.city_map.traffic_density.copy()

    def query_env_with_task(self, task_description: str) -> str:
        """根据任务描述查询相关的环境信息"""
        env_info = []
        
        # 1. 查找相关事件
        active_events = []
        for event_id, event in self._env_data["events"].items():
            if event["is_active"]:
                event_desc = (f"{event['type']} incident at location {event['location']}, "
                            f"floor {event['floor']}, severity {event['severity']}, "
                            f"casualties: {event['casualties']}")
                active_events.append(event_desc)
        
        if active_events:
            env_info.append("Active emergencies:\n" + "\n".join(active_events))
        
        # 2. 查找相关部门信息
        task_lower = task_description.lower()
        relevant_departments = []
        
        # 根据任务描述确定相关部门
        dept_keywords = {
            "fire_department": ["fire", "rescue", "hazmat", "evacuation"],
            "police_department": ["police", "security", "traffic", "crowd", "evacuation"],
            "medical_department": ["medical", "ambulance", "hospital", "injury", "casualty"]
        }
        
        for dept_name, keywords in dept_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                dept = self._env_data["departments"][dept_name]
                # 获取部门位置信息
                locations = [f"- {loc['location']}" for loc in dept["locations"]]
                # 获取在岗人员信息
                on_duty_count = len(dept["on_duty"])
                # 获取可用资源信息
                personnel_count = sum(len(staff) for staff in dept["personnel"].values())
                
                dept_desc = [
                    f"{dept_name.replace('_', ' ').title()}:",
                    f"Locations:\n" + "\n".join(locations),
                    f"Personnel on duty: {on_duty_count}/{personnel_count}",
                    f"Functions: {', '.join(dept['functions'])}",
                    f"Emergency contact: {dept['contacts']['emergency']}"
                ]
                relevant_departments.append("\n".join(dept_desc))
        
        if relevant_departments:
            env_info.append("Relevant departments:\n" + "\n\n".join(relevant_departments))
        
        # 3. 查找相关资源
        available_resources = []
        for resource_type, units in self._env_data["resources"].items():
            if isinstance(units, list):
                available = len([u for u in units if isinstance(u, tuple) and len(u) > 2 and u[2] == "available"])
            else:
                available = units if isinstance(units, (int, float)) else 0
            available_resources.append(f"{resource_type}: {available} units available")
        
        if available_resources:
            env_info.append("Available resources:\n" + "\n".join(available_resources))
        
        # 4. 查找相关建筑
        relevant_buildings = []
        for building_id, building in self._env_data["buildings"].items():
            if any(facility in task_lower for facility in 
                  ["hospital", "fire", "police", "residential", "commercial"]):
                building_desc = (f"{building['type']} building at {building['location']}, "
                               f"{building['floors']} floors")
                relevant_buildings.append(building_desc)
        
        if relevant_buildings:
            env_info.append("Relevant facilities:\n" + "\n".join(relevant_buildings))
        
        # 5. 交通状态
        if self._env_data["traffic"] is not None:
            avg_traffic = np.mean(self._env_data["traffic"])
            traffic_status = "heavy" if avg_traffic > 0.7 else "moderate" if avg_traffic > 0.3 else "light"
            env_info.append(f"Current traffic condition: {traffic_status}")
        
        return "\n\n".join(env_info)

    def query_history(self, name: str) -> str:
        """查询智能体的历史记录"""
        if name not in self._history_data:
            return "No history available."
        return self._history_data[name]

    def query_other_agent_state(self, name: str) -> List[str]:
        """查询其他智能体的状态"""
        other_agents = []
        for agent_name, history in self._history_data.items():
            if agent_name != name:
                other_agents.append(f"Agent {agent_name}: {history}")
        return other_agents

    def query_agent(self, name) -> str:
        # used by controller
        for item in self._agent_data:
            if item["name"] == name:
                return item["content"]
        return "No agent found."

    def query_agent_list(self, name_list: list) -> [str]:
        # used by controller
        result_list = []
        for name in name_list:
            result_list.append(self.query_agent(name))

        return result_list

    def update_database_init(self, info: list):
        self._logger.debug("=" * 20 + " Update Database Init " + "=" * 20)
        self._logger.info(f"gathering info data: \n{info}")
        
        new_info = info.copy()
        for item in new_info:
            item["status"] = item["message"] if item["status"] else {}

        for item in new_info:
            # 处理环境数据
            env_data = self._process_env(item)
            if env_data:
                # 更新部门信息
                if "departments" in env_data:
                    for dept_name, dept_data in env_data["departments"].items():
                        if dept_name in self._env_data["departments"]:
                            self._env_data["departments"][dept_name].update(dept_data)
                
                # 更新建筑信息
                if "buildings" in env_data:
                    self._env_data["buildings"].update(env_data["buildings"])
                
                # 更新资源信息
                if "resources" in env_data:
                    self._env_data["resources"].update(env_data["resources"])
                
                # 更新事件信息
                if "events" in env_data:
                    self._env_data["events"].update(env_data["events"])
                
                # 更新交通信息
                if "traffic" in env_data:
                    self._env_data["traffic"] = env_data["traffic"]
            
            # 处理智能体数据
            agent_data = self._process_agent(item)
            if agent_data:
                # 更新智能体数据
                for i, data in enumerate(self._agent_data):
                    if data["name"] == agent_data["name"]:
                        self._agent_data.pop(i)
                        break
                self._agent_data.append(agent_data)
                self._logger.info(f"Update agent {agent_data['name']} successfully")

        self._logger.info("Update database finished")

    def update_database(self, new_info: dict):
        self._logger.info("Start updating database...")
        new_info["status"] = new_info["status"]["message"] if new_info["status"]["status"] else {}

        # 处理历史记录
        history = self._process_history(new_info)
        if history:
            self._history_data.update(history)
        
        # 处理环境数据
        env_data = self._process_env(new_info)
        if env_data:
            # 更新部门信息
            if "departments" in env_data:
                for dept_name, dept_data in env_data["departments"].items():
                    if dept_name in self._env_data["departments"]:
                        self._env_data["departments"][dept_name].update(dept_data)
            
            # 更新建筑信息
            if "buildings" in env_data:
                self._env_data["buildings"].update(env_data["buildings"])
            
            # 更新资源信息
            if "resources" in env_data:
                self._env_data["resources"].update(env_data["resources"])
            
            # 更新事件信息
            if "events" in env_data:
                self._env_data["events"].update(env_data["events"])
            
            # 更新交通信息
            if "traffic" in env_data:
                self._env_data["traffic"] = env_data["traffic"]
            
            self._logger.info("Update environment data successfully")
        
        # 处理智能体数据
        agent_data = self._process_agent(new_info)
        if agent_data:
            # 更新智能体数据
            for i, data in enumerate(self._agent_data):
                if data["name"] == agent_data["name"]:
                    self._agent_data.pop(i)
                    break
            self._agent_data.append(agent_data)
            self._logger.info(f"Update agent {agent_data['name']} successfully")

        self._logger.info("Update database finished")

    def _process_env(self, item: dict) -> dict:
        """
        Process environment data from input item.
        
        Args:
            item (dict): Input item containing environment data
            
        Returns:
            dict: Processed environment data
        """
        try:
            env_data = {}
            
            # Extract departments data if available
            if "departments" in item:
                env_data["departments"] = item["departments"]
            
            # Extract buildings data if available
            if "buildings" in item:
                env_data["buildings"] = item["buildings"]
                
            # Extract resources data if available
            if "resources" in item:
                env_data["resources"] = item["resources"]
                
            # Extract traffic data if available
            if "traffic" in item:
                env_data["traffic"] = item["traffic"]
                
            return env_data
        except Exception as e:
            self._logger.error(f"Error processing environment data: {str(e)}")
            return {}

    def _process_agent(self, item: dict) -> dict:
        """
        Process agent data from input item.
        
        Args:
            item (dict): Input item containing agent data
            
        Returns:
            dict: Processed agent data
        """
        try:
            agent_data = {
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "status": item.get("status", {}),
                "location": item.get("location", None),
                "resources": item.get("resources", {}),
                "assignments": item.get("assignments", []),
                "current_task": item.get("current_task", None),
                "history": item.get("history", [])
            }
            
            # Add department-specific data based on agent type
            if "type" in item:
                if item["type"] == "medical":
                    agent_data["medical_team"] = item.get("medical_team", [])
                    agent_data["hospital_id"] = item.get("hospital_id", None)
                elif item["type"] == "rescue":
                    agent_data["rescue_team"] = item.get("rescue_team", [])
                    agent_data["equipment"] = item.get("equipment", [])
                elif item["type"] == "security":
                    agent_data["security_team"] = item.get("security_team", [])
                    agent_data["patrol_area"] = item.get("patrol_area", None)
                elif item["type"] == "monitor":
                    agent_data["monitoring_area"] = item.get("monitoring_area", None)
                    agent_data["sensor_data"] = item.get("sensor_data", {})
                elif item["type"] == "traffic":
                    agent_data["controlled_roads"] = item.get("controlled_roads", [])
                    agent_data["traffic_status"] = item.get("traffic_status", {})
            
            return agent_data
        except Exception as e:
            self._logger.error(f"Error processing agent data: {str(e)}")
            return {}

    def _process_history(self, item: dict) -> dict:
        """
        Process history data from input item.
        
        Args:
            item (dict): Input item containing history data
            
        Returns:
            dict: Processed history data
        """
        try:
            history_data = {}
            
            # 提取代理名称
            if "name" not in item:
                return {}
            agent_name = item["name"]
            
            # 初始化代理的历史记录
            if agent_name not in self._history_data:
                self._history_data[agent_name] = []
            
            # 创建历史记录条目
            history_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task": item.get("task", ""),
                "feedback": item.get("feedback", {}),
                "status": item.get("status", {})
            }
            
            # 添加到历史记录
            history_data[agent_name] = self._history_data[agent_name] + [history_entry]
            
            return history_data
        except Exception as e:
            self._logger.error(f"Error processing history data: {str(e)}")
            return {}
