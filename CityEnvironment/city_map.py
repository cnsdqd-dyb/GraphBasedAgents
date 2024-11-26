from typing import Dict, List, Tuple
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Building:
    id: str
    type: str  # 'hospital', 'fire_station', 'police_station', 'residential', 'commercial'
    location: Tuple[float, float]  # x, y coordinates
    floors: int
    capacity: int  # 容纳人数或资源数量
    resources: Dict  # 不同建筑类型有不同的资源

class CityMap:
    def __init__(self, size: Tuple[int, int] = (1000, 1000)):
        self.size = size
        self.buildings: Dict[str, Building] = {}
        self.roads = np.zeros(size)  # 0: no road, 1: road
        self.traffic_density = np.zeros(size)  # 0-1: traffic density
        self.population_density = np.zeros(size)  # 人口密度
        
    def add_building(self, building: Building):
        self.buildings[building.id] = building
        
    def get_nearest_building(self, location: Tuple[float, float], building_type: str = None) -> Tuple[str, float]:
        """获取最近的指定类型建筑"""
        min_dist = float('inf')
        nearest_id = None
        
        for bid, building in self.buildings.items():
            if building_type and building.type != building_type:
                continue
            dist = ((building.location[0] - location[0])**2 + 
                   (building.location[1] - location[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_id = bid
                
        return nearest_id, min_dist
        
    def get_path_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """获取考虑道路和交通的实际距离"""
        direct_distance = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
        # 考虑交通密度的影响
        avg_traffic = np.mean(self.traffic_density[
            int(min(start[0], end[0])):int(max(start[0], end[0])),
            int(min(start[1], end[1])):int(max(start[1], end[1]))
        ])
        return direct_distance * (1 + avg_traffic)
        
    def get_travel_time(self, start: Tuple[float, float], end: Tuple[float, float], 
                       speed: float = 50.0) -> float:
        """估算行驶时间（分钟）"""
        distance = self.get_path_distance(start, end)
        return distance / speed * 60  # 转换为分钟
        
    def update_traffic(self, decay_factor: float = 0.95):
        """更新交通状况"""
        self.traffic_density *= decay_factor
        # 随机增加一些交通流量
        self.traffic_density += np.random.uniform(0, 0.1, self.size)
        self.traffic_density = np.clip(self.traffic_density, 0, 1)

def create_default_city() -> CityMap:
    """创建一个默认的城市地图"""
    city = CityMap()
    
    # 添加医院
    hospitals = [
        Building("hospital_1", "hospital", (100, 100), 10, 500, 
                {"beds": 200, "ambulances": 10, "doctors": 50, "nurses": 100}),
        Building("hospital_2", "hospital", (800, 800), 15, 800,
                {"beds": 300, "ambulances": 15, "doctors": 80, "nurses": 160}),
    ]
    
    # 添加消防站
    fire_stations = [
        Building("fire_1", "fire_station", (200, 200), 3, 50,
                {"trucks": 5, "firefighters": 30, "equipment": 100}),
        Building("fire_2", "fire_station", (700, 700), 3, 50,
                {"trucks": 5, "firefighters": 30, "equipment": 100}),
    ]
    
    # 添加警察局
    police_stations = [
        Building("police_1", "police_station", (300, 300), 5, 100,
                {"cars": 20, "officers": 100, "equipment": 200}),
        Building("police_2", "police_station", (600, 600), 5, 100,
                {"cars": 20, "officers": 100, "equipment": 200}),
    ]
    
    # 添加住宅和商业建筑
    for i in range(10):
        residential = Building(
            f"residential_{i}", "residential",
            (random.uniform(0, 1000), random.uniform(0, 1000)),
            random.randint(5, 30), random.randint(100, 500),
            {"residents": random.randint(100, 500)}
        )
        commercial = Building(
            f"commercial_{i}", "commercial",
            (random.uniform(0, 1000), random.uniform(0, 1000)),
            random.randint(5, 50), random.randint(200, 1000),
            {"workers": random.randint(50, 200)}
        )
        city.add_building(residential)
        city.add_building(commercial)
    
    # 添加所有应急建筑
    for building in hospitals + fire_stations + police_stations:
        city.add_building(building)
    
    # 初始化一些道路和交通
    for i in range(0, 1000, 100):
        city.roads[i:i+10, :] = 1  # 水平道路
        city.roads[:, i:i+10] = 1  # 垂直道路
    
    return city
