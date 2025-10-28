import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent.parent))

from pipeline import Pipeline

def main():
    # Инициализация пайплайна с именем робота из конфига
    pipeline = Pipeline(robot_name="rozum_pulse_75")
    
    # Задание целевых точек в пространстве [x, y, z]
    pipeline.set_targets([[-0.4, 0.4, 0.7], [-0.1, 0.4, 0.7], [0.3, 0.4, 0.7]])
    
    # Планирование пути с помощью RRT*
    pipeline.plan_path()
    
    # Запуск симуляции движения
    pipeline.run_simulation()

if __name__ == "__main__":
    main()