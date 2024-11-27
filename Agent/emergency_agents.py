from typing import Dict, List, Tuple
from .abstract_agent import AbstractAgent
from CityEnvironment.city_emergency_env import CityEmergencyEnv
from langchain.agents import tool

class MedicalRescueAgent(AbstractAgent):
    def __init__(self, name, env: CityEmergencyEnv, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        self.env = env
        self.setup_tools()
        
    def setup_tools(self):
        self.tools = [
            self.get_medical_resources,
            self.organize_medical_team,
            self.create_rescue_plan
        ]
        
    @tool
    def get_medical_resources(self) -> Dict:
        """Get information about available medical resources"""
        status, data, desc = self.env.get_building_info("hospital")
        return {
            "status": status,
            "data": data if status else {},
            "desc": desc
        }
        
    @tool
    def organize_medical_team(self, hospital_id: str, team_size: int) -> Dict:
        """Organize a medical rescue team from a specific hospital"""
        # 获取医院信息
        status, hospitals, desc = self.env.get_building_info("hospital")
        if not status or hospital_id not in hospitals:
            return {
                "status": False,
                "data": {},
                "desc": "Hospital not found"
            }
        
        hospital = hospitals[hospital_id]
        if hospital.resources["doctors"] < team_size:
            return {
                "status": False,
                "data": {},
                "desc": "Insufficient medical personnel"
            }
            
        # 部署救护车
        status, deployment, desc = self.env.deploy_resource(
            "ambulances", f"{hospital_id}_ambulance_1",
            hospital.location, self.env.active_events[0].location
        )
        
        if not status:
            return {
                "status": False,
                "data": {},
                "desc": f"Failed to deploy ambulance: {desc}"
            }
            
        return {
            "status": True,
            "data": {
                "team": {
                    "hospital": hospital_id,
                    "size": team_size,
                    "ambulance": deployment
                }
            },
            "desc": f"Successfully organized medical team from {hospital_id}"
        }
        
    @tool
    def create_rescue_plan(self, event_id: str) -> Dict:
        """Create a medical rescue plan based on the situation"""
        # 获取事件信息
        status, events, desc = self.env.get_event_info(event_id)
        if not status:
            return {
                "status": False,
                "data": {},
                "desc": f"Event not found: {desc}"
            }
            
        event = events[event_id]
        
        # 获取最近的医院
        status, hospitals, desc = self.env.get_building_info("hospital")
        if not status:
            return {
                "status": False,
                "data": {},
                "desc": f"Failed to get hospital information: {desc}"
            }
            
        # 找到最近的医院
        nearest_hospital = min(
            hospitals.items(),
            key=lambda x: ((x[1].location[0] - event["location"][0])**2 + 
                         (x[1].location[1] - event["location"][1])**2)**0.5
        )
        
        # 根据事件严重程度决定资源分配
        required_resources = {
            "high": {"ambulances": 3, "doctors": 5, "nurses": 10},
            "medium": {"ambulances": 2, "doctors": 3, "nurses": 6},
            "low": {"ambulances": 1, "doctors": 2, "nurses": 4}
        }[event["severity"]]
        
        return {
            "status": True,
            "data": {
                "plan": {
                    "primary_hospital": nearest_hospital[0],
                    "required_resources": required_resources,
                    "estimated_response_time": self.env.city_map.get_travel_time(
                        nearest_hospital[1].location,
                        event["location"]
                    ),
                    "backup_hospitals": [
                        h for h in hospitals if h != nearest_hospital[0]
                    ][:2]
                }
            },
            "desc": f"Created rescue plan with {nearest_hospital[0]} as primary hospital"
        }
        
    def run(self, instruction: str, **kwargs):
        """Run the medical rescue agent with given instruction"""
        # 使用LangChain来处理指令并调用相应的工具
        pass

class EmergencyRescueAgent(AbstractAgent):
    def __init__(self, name, env: CityEmergencyEnv, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        self.env = env
        self.setup_tools()
        
    def setup_tools(self):
        self.tools = [
            self.identify_hazard,
            self.organize_rescue_team,
            self.create_rescue_plan
        ]
        
    @tool
    def identify_hazard(self, location: tuple) -> Dict:
        """Identify hazard information at given location"""
        status, env_data, desc = self.env.get_environmental_data()
        if not status:
            return {"success": False, "error": desc}
            
        # Analyze hazard based on environmental data
        hazard_level = "HIGH" if env_data["gas_concentration"] > 10.0 else "MEDIUM" if env_data["gas_concentration"] > 5.0 else "LOW"
        
        return {
            "success": True,
            "hazard_info": {
                "type": "gas_leak",
                "level": hazard_level,
                "environmental_data": env_data
            }
        }
        
    @tool
    def organize_rescue_team(self, team_type: str, size: int) -> Dict:
        """Organize specific type of rescue team"""
        status, units, desc = self.env.get_rescue_units_info()
        if not status:
            return {"success": False, "error": desc}
            
        available_units = [u for u in units.items() if u[1]["type"] == team_type and u[1]["personnel"] >= size]
        if not available_units:
            return {"success": False, "error": f"No available {team_type} units with sufficient personnel"}
            
        return {
            "success": True,
            "team": {
                "unit_id": available_units[0][0],
                "type": team_type,
                "size": size
            }
        }
        
    @tool
    def create_rescue_plan(self, hazard_type: str, location: tuple) -> Dict:
        """Create a rescue plan based on hazard type and location"""
        # Get environmental data for planning
        env_status, env_data, env_desc = self.env.get_environmental_data()
        if not env_status:
            return {"success": False, "error": env_desc}
            
        # Get available rescue units
        units_status, units, units_desc = self.env.get_rescue_units_info()
        if not units_status:
            return {"success": False, "error": units_desc}
            
        return {
            "success": True,
            "plan": {
                "hazard_type": hazard_type,
                "location": location,
                "required_teams": [
                    {"type": "firefighters", "size": 5},
                    {"type": "hazmat", "size": 3}
                ],
                "evacuation_radius": 200 if env_data["gas_concentration"] > 10.0 else 100
            }
        }

class SecurityControlAgent(AbstractAgent):
    def __init__(self, name, env: CityEmergencyEnv, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        self.env = env
        self.setup_tools()
        
    def setup_tools(self):
        self.tools = [
            self.set_security_perimeter,
            self.create_evacuation_plan,
            self.deploy_security_personnel
        ]
        
    @tool
    def set_security_perimeter(self, center: tuple, radius: float) -> Dict:
        """Set up security perimeter around incident"""
        status, result, desc = self.env.set_danger_zone([(center[0], center[1], radius)])
        if not status:
            return {"success": False, "error": desc}
            
        return {
            "success": True,
            "perimeter": {
                "center": center,
                "radius": radius
            }
        }
        
    @tool
    def create_evacuation_plan(self, danger_zones: List[tuple]) -> Dict:
        """Create evacuation plan based on danger zones"""
        # Get traffic information for planning routes
        status, roads, desc = self.env.get_traffic_info()
        if not status:
            return {"success": False, "error": desc}
            
        return {
            "success": True,
            "evacuation_plan": {
                "zones": danger_zones,
                "assembly_points": [(x + 300, y + 300) for x, y, _ in danger_zones],
                "routes": []  # Would be populated based on road network
            }
        }
        
    @tool
    def deploy_security_personnel(self, locations: List[tuple]) -> Dict:
        """Deploy security personnel to specific locations"""
        return {
            "success": True,
            "deployment": {
                "locations": locations,
                "personnel_count": len(locations) * 2
            }
        }

class DisasterMonitoringAgent(AbstractAgent):
    def __init__(self, name, env: CityEmergencyEnv, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        self.env = env
        self.setup_tools()
        
    def setup_tools(self):
        self.tools = [
            self.get_environmental_data,
            self.analyze_risk,
            self.predict_disaster_spread
        ]
        
    @tool
    def get_environmental_data(self) -> Dict:
        """Get current environmental data"""
        status, data, desc = self.env.get_environmental_data()
        if not status:
            return {"success": False, "error": desc}
            
        return {"success": True, "data": data}
        
    @tool
    def analyze_risk(self, env_data: Dict) -> Dict:
        """Analyze risk based on environmental data"""
        if "gas_concentration" not in env_data:
            return {"success": False, "error": "Invalid environmental data"}
            
        risk_level = "HIGH" if env_data["gas_concentration"] > 10.0 else \
                    "MEDIUM" if env_data["gas_concentration"] > 5.0 else "LOW"
                    
        return {
            "success": True,
            "risk_assessment": {
                "level": risk_level,
                "factors": {
                    "gas_concentration": env_data["gas_concentration"],
                    "wind_speed": env_data["wind_speed"]
                }
            }
        }
        
    @tool
    def predict_disaster_spread(self, current_data: Dict) -> Dict:
        """Predict how the disaster might spread"""
        if "wind_direction" not in current_data or "wind_speed" not in current_data:
            return {"success": False, "error": "Missing wind data"}
            
        # Simple prediction based on wind
        spread_direction = current_data["wind_direction"]
        spread_speed = current_data["wind_speed"] * 0.5
        
        return {
            "success": True,
            "prediction": {
                "direction": spread_direction,
                "speed": spread_speed,
                "affected_area_growth": f"{spread_speed * 10:.1f} meters per minute"
            }
        }

class TrafficControlAgent(AbstractAgent):
    def __init__(self, name, env: CityEmergencyEnv, local_port=5000, model="gpt-4-1106-preview", api_key_list=[]):
        super().__init__(name, local_port, model, api_key_list)
        self.env = env
        self.setup_tools()
        
    def setup_tools(self):
        self.tools = [
            self.get_traffic_status,
            self.plan_rescue_route,
            self.implement_traffic_control
        ]
        
    @tool
    def get_traffic_status(self) -> Dict:
        """Get current traffic status"""
        status, roads, desc = self.env.get_traffic_info()
        if not status:
            return {"success": False, "error": desc}
            
        return {"success": True, "traffic_data": roads}
        
    @tool
    def plan_rescue_route(self, start: tuple, end: tuple) -> Dict:
        """Plan optimal rescue route between two points"""
        status, roads, desc = self.env.get_traffic_info()
        if not status:
            return {"success": False, "error": desc}
            
        # Simple direct route for now
        return {
            "success": True,
            "route": {
                "start": start,
                "end": end,
                "distance": ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5,
                "estimated_time": "10 minutes"
            }
        }
        
    @tool
    def implement_traffic_control(self, road_ids: List[str]) -> Dict:
        """Implement traffic control on specified roads"""
        status, roads, desc = self.env.get_traffic_info()
        if not status:
            return {"success": False, "error": desc}
            
        controlled_roads = [r for r in road_ids if r in roads]
        
        return {
            "success": True,
            "traffic_control": {
                "controlled_roads": controlled_roads,
                "status": "implemented",
                "affected_areas": len(controlled_roads)
            }
        }
