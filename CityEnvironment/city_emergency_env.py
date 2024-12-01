from typing import Dict, List, Tuple
from Env.abstract_env import MultiAgentEnvironment
import random
from datetime import datetime, timedelta
from CityEnvironment.city_map import CityMap, create_default_city, Building
from Agent.emergency_agents import Agent
import numpy as np

class EmergencyEvent:
    def __init__(self, event_type: str, location: Tuple[float, float], floor: int,
                 start_time: datetime, severity: str, properties: Dict):
        self.type = event_type
        self.location = location
        self.floor = floor
        self.start_time = start_time
        self.severity = severity  # 'low', 'medium', 'high'
        self.properties = properties
        self.is_active = True
        self.casualties = 0
        self.affected_radius = 0
        
    def update(self, elapsed_minutes: float):
        """更新事件状态"""
        if self.type == "fire":
            # 火势蔓延
            self.affected_radius += 0.5 * elapsed_minutes  # 每分钟扩大0.5米
            # 更新伤亡
            if self.severity == "high":
                self.casualties += random.randint(0, 2)
            elif self.severity == "medium":
                self.casualties += random.randint(0, 1)

class Resource:
    def __init__(self, resource_id: str, resource_type: str, location: Tuple[float, float], 
                 status: str = "available", properties: Dict = None):
        self.id = resource_id
        self.type = resource_type  # vehicle, equipment, personnel
        self.location = location
        self.status = status  # available, in_use, maintenance, offline
        self.properties = properties or {}
        self.assigned_to = None  # agent_id that this resource is assigned to

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "location": self.location,
            "status": self.status,
            "properties": self.properties,
            "assigned_to": self.assigned_to
        }

class CityEmergencyEnv(MultiAgentEnvironment):
    """City emergency environment simulator"""
    
    def __init__(self, city_size=(10, 10), num_hospitals=2, num_fire_stations=2, 
                 num_police_stations=2, population_density=0.7, traffic_density=0.5,
                 task_id=0, host="0.0.0.0", port=25565, task_name="emergency_response",
                 virtual_debug=False):
        """初始化城市环境"""
        super().__init__("city_emergency", task_id, host, port, task_name, virtual_debug)
        
        # 创建城市地图
        self.city_map = create_default_city(
            size=city_size,
            num_hospitals=num_hospitals,
            num_fire_stations=num_fire_stations,
            num_police_stations=num_police_stations,
            population_density=population_density,
            traffic_density=traffic_density
        )
        print(f"Created city map with {len(self.city_map.buildings)} buildings")
        
        # 初始化时间和事件
        self.current_time = datetime.now()
        self.active_events: Dict[str, EmergencyEvent] = {}
        
        # 初始化资源管理
        self.resources: Dict[str, Resource] = {}
        self.agent_resources: Dict[str, List[str]] = {}  # agent_id -> list of resource_ids
        self.resource_assignments: Dict[str, List[str]] = {}  # agent_id -> list of resource_ids
        
        # 初始化基础资源
        self._initialize_resources()
        
    def _initialize_resources(self):
        """初始化城市环境中的基础资源"""
        # 初始化救护车
        for i in range(5):
            resource_id = f"ambulance_{i}"
            location = self._get_building_location("hospital")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="vehicle",
                location=location,
                properties={"vehicle_type": "ambulance", "capacity": 2}
            )
            
        # 初始化消防车
        for i in range(3):
            resource_id = f"fire_truck_{i}"
            location = self._get_building_location("fire_station")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="vehicle",
                location=location,
                properties={"vehicle_type": "fire_truck", "water_capacity": 1000}
            )
            
        # 初始化医务人员
        for i in range(10):
            resource_id = f"medic_{i}"
            location = self._get_building_location("hospital")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="personnel",
                location=location,
                properties={"role": "medic", "skill_level": random.randint(1, 5)}
            )
            
        # 初始化消防员
        for i in range(8):
            resource_id = f"firefighter_{i}"
            location = self._get_building_location("fire_station")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="personnel",
                location=location,
                properties={"role": "firefighter", "skill_level": random.randint(1, 5)}
            )
            
        # 初始化医疗设备
        for i in range(15):
            resource_id = f"med_equipment_{i}"
            location = self._get_building_location("hospital")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="equipment",
                location=location,
                properties={"equipment_type": "medical", "condition": "good"}
            )

        # 初始化警车
        for i in range(4):
            resource_id = f"police_car_{i}"
            location = self._get_building_location("police_station")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="vehicle",
                location=location,
                properties={"vehicle_type": "police_car", "capacity": 4}
            )
            
        # 初始化警察
        for i in range(12):
            resource_id = f"police_{i}"
            location = self._get_building_location("police_station")
            self.resources[resource_id] = Resource(
                resource_id=resource_id,
                resource_type="personnel",
                location=location,
                properties={"role": "police", "skill_level": random.randint(1, 5)}
            )

    def _get_building_location(self, building_type: str) -> Tuple[float, float]:
        """获取随机建筑位置"""
        buildings = [b for b in self.city_map.buildings.values() if b.type == building_type]
        print(f"Found {len(buildings)} buildings of type {building_type}")
        print(f"All building types: {[b.type for b in self.city_map.buildings.values()]}")
        if buildings:
            building = random.choice(buildings)
            return building.location
        print(f"Warning: No buildings found of type {building_type}, returning default location (0, 0)")
        return (0, 0)

    def assign_resource(self, resource_id: str, agent_id: str) -> bool:
        """将资源分配给特定的agent"""
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        if resource.status != "available":
            return False
            
        if agent_id not in self.resource_assignments:
            self.resource_assignments[agent_id] = []
            
        resource.status = "in_use"
        resource.assigned_to = agent_id
        self.resource_assignments[agent_id].append(resource_id)
        return True
        
    def release_resource(self, resource_id: str) -> bool:
        """释放资源"""
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        if resource.assigned_to:
            self.resource_assignments[resource.assigned_to].remove(resource_id)
            resource.assigned_to = None
            resource.status = "available"
        return True
        
    def update_resource_status(self, resource_id: str, new_status: str, 
                             new_location: Tuple[float, float] = None) -> bool:
        """更新资源状态和位置"""
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        resource.status = new_status
        if new_location:
            resource.location = new_location
        return True
        
    def get_agent_resources(self, agent_id: str) -> List[Dict]:
        """获取分配给特定agent的所有资源状态"""
        if agent_id not in self.resource_assignments:
            return []
            
        return [self.resources[rid].to_dict() 
                for rid in self.resource_assignments[agent_id]]
                
    def init_emergency_scenario(self, scenario_type: str, location: Tuple[float, float] = None,
                              floor: int = None, severity: str = "medium") -> Tuple[bool, Dict, str]:
        """初始化突发事件场景"""
        try:
            if not location:
                # 随机选择一个建筑
                building = random.choice(list(self.city_map.buildings.values()))
                location = building.location
                floor = random.randint(1, building.floors)
            
            event_id = f"{scenario_type}_{len(self.active_events)}"
            
            if scenario_type == "fire":
                properties = {
                    "temperature": random.uniform(400, 800),
                    "smoke_level": random.uniform(0.5, 1.0),
                    "spread_rate": random.uniform(0.3, 0.8)
                }
            elif scenario_type == "gas_leak":
                properties = {
                    "gas_concentration": random.uniform(5.0, 15.0),
                    "leak_rate": random.uniform(0.1, 0.5),
                    "explosive_risk": random.uniform(0.3, 0.8)
                }
            else:
                return False, {}, f"Unknown scenario type: {scenario_type}"
                
            event = EmergencyEvent(scenario_type, location, floor,
                                 self.current_time, severity, properties)
            self.active_events[event_id] = event
            
            return True, {
                "event_id": event_id,
                "location": location,
                "floor": floor,
                "properties": properties
            }, f"Successfully initialized {scenario_type} scenario"
            
        except Exception as e:
            return False, {}, f"Failed to initialize scenario: {str(e)}"
            
    def update_environment(self, elapsed_minutes: float = 1.0) -> Tuple[bool, Dict, str]:
        """更新环境状态"""
        try:
            self.current_time += timedelta(minutes=elapsed_minutes)
            self.city_map.update_traffic()
            
            # 更新所有活动事件
            for event in self.active_events.values():
                event.update(elapsed_minutes)
            
            # 更新资源使用情况
            current_resources = {
                "ambulances": self._count_available_resources("ambulances"),
                "fire_trucks": self._count_available_resources("fire_trucks"),
                "police_cars": self._count_available_resources("police_cars")
            }
            
            return True, {
                "current_time": self.current_time,
                "active_events": len(self.active_events),
                "available_resources": current_resources
            }, "Environment updated successfully"
            
        except Exception as e:
            return False, {}, f"Failed to update environment: {str(e)}"
    
    def _count_available_resources(self, resource_type: str) -> int:
        """统计可用资源数量"""
        total = sum(b.resources.get(resource_type, 0) 
                   for b in self.city_map.buildings.values() 
                   if resource_type in b.resources)
        in_use = len(self.deployed_resources[resource_type])
        return total - in_use
        
    def deploy_resource(self, resource_type: str, unit_id: str, 
                       from_location: Tuple[float, float],
                       to_location: Tuple[float, float]) -> Tuple[bool, Dict, str]:
        """部署资源到指定位置"""
        try:
            if self._count_available_resources(resource_type) <= 0:
                return False, {}, f"No available {resource_type}"
                
            travel_time = self.city_map.get_travel_time(from_location, to_location)
            
            self.deployed_resources[resource_type].append((unit_id, to_location, "en_route"))
            self.resource_usage[resource_type][unit_id] = {
                "start_time": self.current_time,
                "estimated_arrival": self.current_time + timedelta(minutes=travel_time)
            }
            
            return True, {
                "unit_id": unit_id,
                "travel_time": travel_time,
                "estimated_arrival": self.current_time + timedelta(minutes=travel_time)
            }, f"Successfully deployed {resource_type}"
            
        except Exception as e:
            return False, {}, f"Failed to deploy resource: {str(e)}"
            
    def get_building_info(self, building_type: str = None) -> Tuple[bool, Dict, str]:
        """获取建筑信息"""
        try:
            buildings = {bid: b for bid, b in self.city_map.buildings.items()
                        if not building_type or b.type == building_type}
            
            return True, buildings, f"Found {len(buildings)} buildings"
        except Exception as e:
            return False, {}, f"Failed to get building information: {str(e)}"
            
    def get_event_info(self, event_id: str = None) -> Tuple[bool, Dict, str]:
        """获取事件信息"""
        try:
            if event_id:
                if event_id not in self.active_events:
                    return False, {}, f"Event {event_id} not found"
                events = {event_id: self.active_events[event_id]}
            else:
                events = self.active_events
                
            event_info = {
                eid: {
                    "type": e.type,
                    "location": e.location,
                    "floor": e.floor,
                    "start_time": e.start_time.isoformat(),
                    "severity": e.severity,
                    "properties": e.properties,
                    "casualties": e.casualties,
                    "affected_radius": e.affected_radius
                } for eid, e in events.items()
            }
            
            return True, event_info, f"Retrieved information for {len(events)} events"
        except Exception as e:
            return False, {}, f"Failed to get event information: {str(e)}"
            
    def get_resource_status(self) -> Tuple[bool, Dict, str]:
        """获取资源使用状态"""
        try:
            status = {
                resource_type: {
                    "total": self._count_available_resources(resource_type) + len(deployed),
                    "available": self._count_available_resources(resource_type),
                    "deployed": len(deployed),
                    "deployment_details": [
                        {
                            "unit_id": unit_id,
                            "location": location,
                            "status": status,
                            "usage_info": self.resource_usage[resource_type].get(unit_id, {})
                        }
                        for unit_id, location, status in deployed
                    ]
                }
                for resource_type, deployed in self.deployed_resources.items()
            }
            
            return True, status, "Successfully retrieved resource status"
        except Exception as e:
            return False, {}, f"Failed to get resource status: {str(e)}"

    def get_environmental_data(self) -> Tuple[bool, Dict, str]:
        """获取环境数据"""
        try:
            # 获取事件和建筑物的影响
            total_risk = 0
            affected_areas = set()
            
            # 检查活跃事件的影响
            for event in self.active_events.values():
                if event.type in ["fire", "chemical_leak", "explosion"]:
                    total_risk += 1
                    affected_areas.add((int(event.location[0]), int(event.location[1])))
            
            # 计算受影响区域的环境数据
            if affected_areas:
                avg_population = np.mean([
                    self.city_map.population_density[x, y] 
                    for x, y in affected_areas
                    if 0 <= x < self.city_map.size[0] and 0 <= y < self.city_map.size[1]
                ])
            else:
                avg_population = 0.1  # 基准人口密度
            
            # 基于事件和人口密度计算环境数据
            base_temp = 25  # 基准温度
            base_gas = 5    # 基准气体浓度
            
            data = {
                "temperature": base_temp + total_risk * 5,  # 每个事件增加5度
                "humidity": max(30, 70 - total_risk * 10),  # 事件降低湿度
                "wind_speed": 5 + total_risk * 2,           # 事件增加风速
                "wind_direction": random.uniform(0, 360),   # 随机风向
                "gas_concentration": base_gas + total_risk * 3,  # 事件增加气体浓度
                "air_quality": max(50, 300 - total_risk * 50),   # 事件降低空气质量
                "visibility": max(2, 10 - total_risk),      # 事件降低能见度
                "population_density": avg_population,
                "active_events": total_risk
            }
            return True, data, "Environmental data retrieved based on current events and population density"
            
        except Exception as e:
            return False, {}, f"Failed to get environmental data: {str(e)}"

    def get_traffic_info(self) -> Tuple[bool, Dict, str]:
        """获取交通信息"""
        try:
            roads = {}
            road_spacing = self.city_map.size[0] // 10
            road_width = max(2, road_spacing // 10)
            
            # 获取道路网格的交通信息
            for i in range(0, self.city_map.size[0], road_spacing):
                road_id = f"road_h_{i}"  # 水平道路
                traffic_density = np.mean(self.city_map.traffic_density[i:i+road_width, :])
                
                # 检查是否有事件影响这条道路
                road_events = []
                for event_id, event in self.active_events.items():
                    if abs(event.location[0] - i) < road_width:
                        road_events.append(event_id)
                
                roads[road_id] = {
                    "traffic_density": float(traffic_density),
                    "average_speed": max(5, 60 * (1 - traffic_density)),  # 基于交通密度计算平均速度
                    "status": "blocked" if road_events else "normal" if traffic_density < 0.5 else "congested",
                    "incidents": len(road_events),
                    "affected_by_events": road_events
                }
                
                road_id = f"road_v_{i}"  # 垂直道路
                traffic_density = np.mean(self.city_map.traffic_density[:, i:i+road_width])
                
                # 检查是否有事件影响这条道路
                road_events = []
                for event_id, event in self.active_events.items():
                    if abs(event.location[1] - i) < road_width:
                        road_events.append(event_id)
                
                roads[road_id] = {
                    "traffic_density": float(traffic_density),
                    "average_speed": max(5, 60 * (1 - traffic_density)),
                    "status": "blocked" if road_events else "normal" if traffic_density < 0.5 else "congested",
                    "incidents": len(road_events),
                    "affected_by_events": road_events
                }
            
            return True, roads, "Traffic information retrieved based on city map and active events"
            
        except Exception as e:
            return False, {}, f"Failed to get traffic information: {str(e)}"

    def get_all_agent_description_tiny(self) -> dict:
        """
        Get brief descriptions of all agents in the environment based on their tools.
        
        Returns:
            dict: A dictionary mapping agent names to their tool descriptions
        """
        descriptions = {}
        for agent in self.agent_pool:
            tool_descriptions = []
            for tool in agent.tools:
                print(tool)
                if hasattr(tool, '__doc__') and tool.__doc__:
                    tool_descriptions.append(f"{tool.name} -- {tool.__doc__.strip()}")
            descriptions[agent.name] = "; ".join(tool_descriptions)
        print(descriptions)
        input()
        return descriptions
        
    def agent_register(self, agent_tools=[], agent_number: int = 1, name_list: [str] = [],
                       model: str = "gpt-4-1106-preview", api_key_list=[]):
        self.logger.warning(
            "[warning but dont worry] agent number not equal to names number, random names will be used")
        assert len(name_list) == agent_number, "agent number not equal to names number"

        for i in range(agent_number):
            agent = Agent(name_list[i], self, tools=agent_tools, model=model, api_key_list=api_key_list)
            self.agent_pool.append(agent)
            self.logger.info(f"Agent {name_list[i]} registered successfully")

    def launch(self, debug: bool = False, fast_api=False):
        """Launch the environment"""
        super().launch(debug, fast_api)
        self.launch_time = datetime.now()
        return True

    def agent_status(self, agent_name: str):  # 返回一个dict
        for agent in self.agent_pool:
            if agent.name == agent_name:
                return Agent.get_status(agent_name)
        return {"message": f"agent {agent_name} not found", "status": False}
    
    def step(self, agent_name: str, action: str, max_turn: int = 2):
        '''
        final_answer, {"input": response["input"], "action_list": action_list, "final_answer": final_answer}
        '''
        self.logger.debug("=" * 20 + " Env Step " + "=" * 20)
        self.logger.info(f"agent {agent_name}")
        self.logger.info("=" * 20 + " Env Step " + "=" * 20)
        find_agent = False
        for agent in self.agent_pool:
            if agent.name == agent_name:
                feedback, detail = agent.run(action, max_turn=max_turn)
                return feedback, detail

        if not find_agent:
            self.logger.warning(f"agent {agent_name} not found")
            return None, {"input": None, "action_list": None, "final_answer": None}

    def stop(self):
        """Stop the environment"""
        self.running = False
        # Clean up resources
        self.active_events.clear()
        self.deployed_resources = {
            "ambulances": [],
            "fire_trucks": [],
            "police_cars": []
        }
        self.resource_usage = {
            "ambulances": {},
            "fire_trucks": {},
            "police_cars": {}
        }
        return True

    def reset(self):
        """Reset the environment to initial state"""
        self.current_time = datetime.now()
        self.active_events.clear()
        self.deployed_resources = {
            "ambulances": [],
            "fire_trucks": [],
            "police_cars": []
        }
        self.resource_usage = {
            "ambulances": {},
            "fire_trucks": {},
            "police_cars": {}
        }
        return True

    def get_init_state(self) -> List[Dict]:
        """Get the initial state of the environment
        
        Returns:
            List[Dict]: List of dictionaries containing information about:
                - Buildings (hospitals, fire stations, police stations, etc.)
                - Available resources
                - Current events
                - Population and traffic density
        """
        init_state = []
        
        # Add building information
        for building_id, building in self.city_map.buildings.items():
            building_info = {
                "type": "building",
                "id": building_id,
                "building_type": building.type,
                "location": building.location,
                "floors": building.floors,
                "capacity": building.capacity,
                "resources": building.resources,
                "status": True,
                "message": {
                    "building_id": building_id,
                    "type": building.type,
                    "location": building.location,
                    "floors": building.floors,
                    "capacity": building.capacity,
                    "resources": building.resources
                }
            }
            init_state.append(building_info)
        
        # Add current events
        for event_id, event in self.active_events.items():
            event_info = {
                "type": "event",
                "id": event_id,
                "event_type": event.type,
                "location": event.location,
                "floor": event.floor,
                "start_time": event.start_time.isoformat(),
                "severity": event.severity,
                "properties": event.properties,
                "casualties": event.casualties,
                "affected_radius": event.affected_radius,
                "status": True,
                "message": {
                    "event_id": event_id,
                    "type": event.type,
                    "location": event.location,
                    "floor": event.floor,
                    "start_time": event.start_time.isoformat(),
                    "severity": event.severity,
                    "properties": event.properties,
                    "casualties": event.casualties,
                    "affected_radius": event.affected_radius
                }
            }
            init_state.append(event_info)
        
        # Add environment status
        env_info = {
            "type": "environment",
            "current_time": self.current_time.isoformat(),
            "city_size": self.city_map.size,
            "deployed_resources": self.deployed_resources,
            "resource_usage": self.resource_usage,
            "status": True,
            "message": {
                "current_time": self.current_time.isoformat(),
                "city_size": self.city_map.size,
                "deployed_resources": self.deployed_resources,
                "resource_usage": self.resource_usage,
                "departments": {
                    "fire_department": {
                        "locations": [b.location for b in self.city_map.buildings.values() if b.type == "fire_station"],
                        "personnel": {"firefighters": [], "commanders": []},
                        "functions": ["fire_control", "rescue", "hazmat_handling"],
                        "on_duty": {},
                        "contacts": {"emergency": "119", "office": "", "email": ""}
                    },
                    "police_department": {
                        "locations": [b.location for b in self.city_map.buildings.values() if b.type == "police_station"],
                        "personnel": {"officers": [], "commanders": []},
                        "functions": ["law_enforcement", "traffic_control", "evacuation"],
                        "on_duty": {},
                        "contacts": {"emergency": "110", "office": "", "email": ""}
                    },
                    "medical_department": {
                        "locations": [b.location for b in self.city_map.buildings.values() if b.type == "hospital"],
                        "personnel": {"doctors": [], "nurses": [], "paramedics": []},
                        "functions": ["emergency_medical_care", "patient_transport"],
                        "on_duty": {},
                        "contacts": {"emergency": "120", "office": "", "email": ""}
                    }
                }
            }
        }
        init_state.append(env_info)
        
        return init_state
