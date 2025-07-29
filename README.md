# Manipulator

Проект по планированию движения роботов-манипуляторов с использованием алгоритмов **RRT (Rapidly-exploring Random Tree)** и **RRT*** в симуляторе MuJoCo.

## Основные компоненты

- **RRT и RRT*** алгоритмы для планирования траекторий без коллизий
- **Симулятор MuJoCo** с реалистичной физикой и визуализацией
- **Обратная кинематика** для достижения целевых позиций

## Установка и настройка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/qrvmil/Manipulator.git
cd Manipulator
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Подготовка конфигурации

1. **Сцена робота** (`simulator/models/your_robot/`):
   - `scene.xml` - основной файл сцены MuJoCo
   - `assets/` - 3D модели и текстуры
   - `objects/` - дополнительные объекты сцены

2. **Конфигурация робота** (`pipeline/config/robots/your_robot.yaml`):
   ```yaml
   name: "robot_name"
   dof: 7  # Количество степеней свободы
   joint_indices: [0, 1, 2, 3, 4, 5, 6]  # Индексы суставов
   joint_limits:  # Ограничения суставов [min, max] в радианах. пример ниже:
     - [-2.96, 2.96]
     - [-2.09, 2.09]
     - [-2.96, 2.96]
     - [-2.09, 2.09]
     - [-2.96, 2.96]
     - [-2.09, 2.09]
     - [-3.05, 3.05]
   
   attachment_site: "ee_site"  # Имя сайта end-effector'а в MuJoCo
   robot_base: "robot_base"    # Имя базового звена робота
   
   mujoco:
     default_scene: "simulator/models/your_robot/scene.xml"
   
   ik_params:
     tolerance: 1e-3      # Точность IK
     max_steps: 500       # Максимальное число итераций
     step_size: 0.005     # Шаг оптимизации
     orientation: [0.0, 1.0, 0.0, 0.0]  # Опционально: целевая ориентация [w, x, y, z]
   
   default_planning:      # Параметры RRT* (опционально)
     rewire_count: 15
     goal_radius: 0.08
     goal_bias: 0.4
     max_iterations: 20000
     step_size: 0.05
     sampling_frequency: 4
   ```

### Запуск планирования и симуляции

1. Создайте скрипт для запуска (пример):
```python
from pipeline import Pipeline

def main():
    # Инициализация пайплайна с именем робота из конфига
    pipeline = Pipeline(robot_name="your_robot")
    
    # Задание целевых точек в пространстве [x, y, z]
    pipeline.set_targets([
        [-0.5, 0.4, 0.7],
        [-0.1, 0.4, 0.7],
        [0.3, 0.4, 0.7]
    ])
    
    # Планирование пути с помощью RRT*
    pipeline.plan_path()
    
    # Запуск симуляции движения
    pipeline.run_simulation()

if __name__ == "__main__":
    main()
```

2. Запустите скрипт:
```bash
python your_script.py
```

## Демонстрация работы

[![image info](https://github.com/qrvmil/Manipulator/blob/main/demo/simple_movement/kuka_rrt.gif)](https://youtu.be/U0jLNqwuZgs)

**[🔗 Смотреть демонстрацию работы RRT](https://youtu.be/U0jLNqwuZgs)**

## Поддерживаемые роботы

- **KUKA iiwa 14** 

## Алгоритмы

### RRT (Rapidly-exploring Random Tree)
- Быстрое исследование конфигурационного пространства
- Вероятностная полнота
- Эффективность в высокоразмерных пространствах

### RRT* (RRT Star) 
- Асимптотическая оптимальность
- Переподключение узлов для улучшения пути
- Сходимость к оптимальному решению

## Структура проекта

```
Manipulator/
├── RRT/                           # Алгоритмы планирования
│   ├── algorithms/
│   │   ├── vanilla_rrt.py        # Базовый RRT
│   │   ├── rrt_star.py           # RRT* с оптимизацией
│   │   └── utils/
│   │       └── node.py           # Узлы дерева
│   └── gifs/                     # Визуализации алгоритмов
├── simulator/                     # MuJoCo симулятор
│   ├── models/                   # Модели роботов
│   │   ├── kuka_iiwa_14/        # KUKA iiwa 14
│   │   │   ├── scene.xml        # Основная сцена
│   │   │   ├── assets/          # 3D модели
│   │   │   └── objects/         # Объекты сцены
│   │   └── vx300/               # ViperX 300
│   └── ik/                      # Обратная кинематика
├── pipeline/                     # Основной пайплайн
│   ├── config/                  # Конфигурационные файлы
│   │   ├── robots/             # Конфиги роботов
│   │   └── simulation_settings.yaml
│   ├── examples/               # Примеры использования
│   └── utils/                 # Вспомогательные функции
├── outputs/                   # Сохраненные траектории
└── demo/                     # Демонстрации
    └── simple_movement/      # Примеры движения
```

## Примечания

- Все сгенерированные траектории сохраняются в директории `outputs/` с уникальными именами
- Временные файлы (например, `qpath.txt`) автоматически удаляются после выполнения
- Для добавления нового робота необходимо создать соответствующие файлы сцены и конфигурации и разместить их в соответствующих директориях (`simulator/models` и `pipeline/config/robots`)  
- Примеры запуска пайплайна с роботом kuka-iiwa14 можно найти в `pipeline/examples`

---

