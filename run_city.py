#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
城市应急响应系统启动文件
"""

from CityEnvironment.city_emergency_env import CityEmergencyEnv
from CityPipe.controller import GlobalController
from CityPipe.data_manager import DataManager
from CityPipe.task_manager import TaskManager
from CityPipe.agent import EmergencyResponseAgent
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
        traffic_density=0.5          # 交通密度
    )

    # 加载API密钥
    api_key_path = os.path.join(os.path.dirname(__file__), "API_KEY_LIST")
    api_key_list = json.load(open(api_key_path, "r"))["OPENAI"]
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
    agent_tools = [
        EmergencyResponseAgent.assess_situation,      # 评估情况
        EmergencyResponseAgent.request_resources,     # 请求资源
        EmergencyResponseAgent.coordinate_response,   # 协调响应
        EmergencyResponseAgent.deploy_units,         # 部署单位
        EmergencyResponseAgent.provide_assistance,   # 提供援助
        EmergencyResponseAgent.report_status,        # 报告状态
        EmergencyResponseAgent.manage_evacuation,    # 管理疏散
        EmergencyResponseAgent.handle_casualties     # 处理伤员
    ]

    # 注册智能体
    agent_names = ["FireChief", "PoliceChief", "MedicalDirector"]
    env.agent_register(
        agent_tools=agent_tools,
        agent_number=len(agent_names),
        name_list=agent_names
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
