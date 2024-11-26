from typing import Dict, List, Tuple
from Env.abstract_env import MultiAgentEnvironment
import random
from datetime import datetime, timedelta
from .city_map import CityMap, create_default_city, Building

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

class CityEmergencyEnv(MultiAgentEnvironment):
    """City emergency environment simulator"""
    
    def __init__(self, task_id, host="0.0.0.0", port=25565, task_name="emergency_response"):
        super().__init__("city_emergency", task_id, host, port, task_name)
        self.city_map = create_default_city()
        self.current_time = datetime.now()
        self.active_events: Dict[str, EmergencyEvent] = {}
        self.deployed_resources = {
            "ambulances": [],  # [(unit_id, location, status), ...]
            "fire_trucks": [],
            "police_cars": []
        }
        self.resource_usage = {
            "ambulances": {},  # {unit_id: {"start_time": datetime, "end_time": datetime}}
            "fire_trucks": {},
            "police_cars": {}
        }
        
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
                    "start_time": e.start_time,
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
