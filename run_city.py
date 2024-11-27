#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
城市应急响应系统启动文件
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from CityEnvironment.city_emergency_env import CityEmergencyEnv
from CityPipe.controller import GlobalController
from CityPipe.data_manager import DataManager
from CityPipe.task_manager import TaskManager
from CityPipe.agent import EmergencyResponseAgent
from Agent.emergency_agents import EmergencyRescueAgent, MedicalRescueAgent, TrafficControlAgent, DisasterMonitoringAgent, SecurityControlAgent
import json
import os

def main():
    # 设置环境
    env = CityEmergencyEnv(
        city_size=(10, 10),          # 城市网格大小
        num_hospitals=2,             # 医院数量
        num_fire_stations=2,         # 消防站数量
        num_police_stations=2,       # 警察局数量
        population_density=0.7,      # 人口密度
        traffic_density=0.5,         # 交通密度
        task_id=0,                   # 任务ID
        task_name="emergency_response" # 任务名称
    )

    # 加载API密钥
    api_key_path = os.path.join(os.path.dirname(__file__), "API_KEY_LIST")
    api_key_list = json.load(open(api_key_path, "r"))["KEY"]
    base_url = "https://api.openai.com/v1"  # OpenAI API基础URL

    # LLM配置
    llm_config = {
        "api_model": "gpt-4-1106-preview",  # 使用GPT-4
        "api_base": base_url,
        "api_key_list": api_key_list
    }

    # 设置智能体模型
    EmergencyResponseAgent.model = "gpt-4-1106-preview"
    EmergencyResponseAgent.base_url = base_url
    EmergencyResponseAgent.api_key_list = api_key_list

    # 定义智能体工具
    emergency_rescue_agent_tools = [
        EmergencyRescueAgent.organize_rescue_team,
        EmergencyRescueAgent.create_rescue_plan,
        EmergencyRescueAgent.identify_hazard,
    ]

    traffic_control_agent_tools = [
        TrafficControlAgent.implement_traffic_control,
        TrafficControlAgent.get_traffic_status,
        TrafficControlAgent.plan_rescue_route
    ]

    disaster_monitoring_agent_tools = [
        DisasterMonitoringAgent.get_environmental_data,
        DisasterMonitoringAgent.analyze_risk,
        DisasterMonitoringAgent.predict_disaster_spread
    ]

    security_control_agent_tools = [
        SecurityControlAgent.create_evacuation_plan,
        SecurityControlAgent.set_security_perimeter,
        SecurityControlAgent.deploy_security_personnel
    ]

    medical_rescue_agent_tools = [
        MedicalRescueAgent.organize_medical_team,
        MedicalRescueAgent.create_rescue_plan,
        MedicalRescueAgent.get_medical_resources
    ]
    # 注册智能体
    agent_names = ["emergency_rescue_agent", "traffic_control_agent", "disaster_monitoring_agent", "security_control_agent", "medical_rescue_agent"]
    agent_tools = [
        emergency_rescue_agent_tools,
        traffic_control_agent_tools,
        disaster_monitoring_agent_tools,
        security_control_agent_tools,
        medical_rescue_agent_tools
    ]

    for i in range(len(agent_names)):
        env.agent_register(
            agent_tools=agent_tools[i],
            agent_number=1,
            name_list=[agent_names[i]]
        )
    # 运行环境
    with env.run():
        # 设置数据管理器
        dm = DataManager(llm=llm_config)
        dm.update_database_init(env.get_init_state())

        # 设置任务管理器
        tm = TaskManager()

        # 设置控制器
        ctrl = GlobalController(llm_config, tm, dm, env)

        # 设置初始任务
        initial_task = """
        监控和响应城市范围内的紧急情况。主要职责包括：
        1. 持续监控潜在的紧急情况
        2. 评估事件的严重程度和影响范围
        3. 协调多部门响应
        4. 优化资源分配
        5. 确保市民安全
        """
        
        scenario_config = {
            "max_events": 3,           # 最大同时事件数
            "event_types": [           # 可能的事件类型
                "fire",
                "traffic_accident",
                "medical_emergency",
                "gas_leak"
            ],
            "response_priorities": {    # 响应优先级
                "life_threatening": 1,
                "property_damage": 2,
                "public_safety": 1,
                "environmental": 2
            }
        }

        tm.init_task(initial_task, scenario_config)

        # 运行控制器
        ctrl.run()

if __name__ == "__main__":
    main()
