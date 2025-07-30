import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent.parent))

from pipeline import Pipeline

def main():
    pipeline = Pipeline(robot_name="ur5e")
    pipeline.set_targets([[-0.4, 0.4, 0.7], [-0.1, 0.4, 0.7], [0.3, 0.4, 0.7]])  
    pipeline.plan_path()  
    pipeline.run_simulation()


if __name__ == "__main__":
    main()