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

def create_default_city(size: Tuple[int, int] = (1000, 1000), 
                       num_hospitals: int = 2,
                       num_fire_stations: int = 2,
                       num_police_stations: int = 2,
                       population_density: float = 0.7,
                       traffic_density: float = 0.5) -> CityMap:
    """创建一个默认的城市地图
    
    Args:
        size: 城市大小 (宽度, 高度)
        num_hospitals: 医院数量
        num_fire_stations: 消防站数量
        num_police_stations: 警察局数量
        population_density: 人口密度 (0-1)
        traffic_density: 交通密度 (0-1)
    """
    city = CityMap(size)
    
    # 添加医院
    hospitals = []
    for i in range(num_hospitals):
        hospital = Building(
            f"hospital_{i+1}", 
            "hospital", 
            (random.uniform(0, size[0]), random.uniform(0, size[1])), 
            random.randint(10, 15), 
            random.randint(500, 800),
            {
                "beds": random.randint(200, 300), 
                "ambulances": random.randint(10, 15), 
                "doctors": random.randint(50, 80), 
                "nurses": random.randint(100, 160)
            }
        )
        hospitals.append(hospital)
    
    # 添加消防站
    fire_stations = []
    for i in range(num_fire_stations):
        fire_station = Building(
            f"fire_{i+1}", 
            "fire_station", 
            (random.uniform(0, size[0]), random.uniform(0, size[1])), 
            3, 
            50,
            {
                "trucks": random.randint(5, 8), 
                "firefighters": random.randint(30, 40), 
                "equipment": random.randint(100, 150)
            }
        )
        fire_stations.append(fire_station)
    
    # 添加警察局
    police_stations = []
    for i in range(num_police_stations):
        police_station = Building(
            f"police_{i+1}", 
            "police_station", 
            (random.uniform(0, size[0]), random.uniform(0, size[1])), 
            5, 
            100,
            {
                "cars": random.randint(15, 25), 
                "officers": random.randint(80, 120), 
                "equipment": random.randint(150, 250)
            }
        )
        police_stations.append(police_station)
    
    # 添加住宅和商业建筑
    num_other_buildings = int(size[0] * size[1] * population_density / 10000)  # 根据人口密度确定建筑数量
    for i in range(num_other_buildings):
        residential = Building(
            f"residential_{i}", 
            "residential",
            (random.uniform(0, size[0]), random.uniform(0, size[1])),
            random.randint(5, 30), 
            random.randint(100, 500),
            {"residents": random.randint(100, 500)}
        )
        commercial = Building(
            f"commercial_{i}", 
            "commercial",
            (random.uniform(0, size[0]), random.uniform(0, size[1])),
            random.randint(5, 50), 
            random.randint(200, 1000),
            {"workers": random.randint(50, 200)}
        )
        city.add_building(residential)
        city.add_building(commercial)
    
    # 添加所有应急建筑
    for building in hospitals + fire_stations + police_stations:
        city.add_building(building)
    
    # 初始化道路和交通
    road_spacing = size[0] // 10  # 每隔1/10的城市大小添加一条道路
    road_width = max(2, road_spacing // 10)  # 道路宽度为间距的1/10，最小为2
    
    for i in range(0, size[0], road_spacing):
        city.roads[i:i+road_width, :] = 1  # 水平道路
        city.roads[:, i:i+road_width] = 1  # 垂直道路
        
        # 设置初始交通密度
        city.traffic_density[i:i+road_width, :] = random.uniform(0, traffic_density)
        city.traffic_density[:, i:i+road_width] = random.uniform(0, traffic_density)
    
    # 设置人口密度
    for building in city.buildings.values():
        x, y = int(building.location[0]), int(building.location[1])
        radius = int(building.capacity ** 0.5)  # 影响半径与建筑容量相关
        for i in range(max(0, x-radius), min(size[0], x+radius)):
            for j in range(max(0, y-radius), min(size[1], y+radius)):
                dist = ((i-x)**2 + (j-y)**2) ** 0.5
                if dist <= radius:
                    city.population_density[i, j] = min(1.0, city.population_density[i, j] + population_density * (1 - dist/radius))
    
    return city
