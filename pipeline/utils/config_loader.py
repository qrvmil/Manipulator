import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class ConfigLoader:

    def __init__(self, config_root: Optional[str] = None):
        if config_root is None:
            current_dir = Path(__file__).parent.parent
            config_root = current_dir / "config"
        
        self.config_root = Path(config_root)
        self.robots_dir = self.config_root / "robots"
        self.scenes_dir = self.config_root / "scenes"
    
    def load_robot_config(self, robot_name: str) -> Dict[str, Any]:
        with open(self.robots_dir / f"{robot_name}.yaml", 'r') as file:
            config = yaml.safe_load(file)

        required_fields = ['name', 'dof', 'joint_limits', 'joint_indices', 'attachment_site', 'mujoco', 'robot_base']

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in robot config '{robot_name}'")
            
        return config

    def load_simulation_config(self) -> Dict[str, Any]:
        with open(self.config_root / "simulation_settings.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        '''
        required_fields = ['timestep', 'solver', 'iterations', 'tolerance']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in simulation config")
        '''
        return config


# Глобальный экземпляр загрузчика конфигураций
config_loader = ConfigLoader()

def get_robot_config(robot_name: str) -> Dict[str, Any]:
    """Удобная функция для загрузки конфигурации робота"""
    return config_loader.load_robot_config(robot_name)

def get_simulation_config() -> Dict[str, Any]:
    """Удобная функция для загрузки настроек симуляции"""
    return config_loader.load_simulation_config()
    