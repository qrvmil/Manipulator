# Manipulator

Проект по планированию движения роботов-манипуляторов с использованием алгоритмов **RRT (Rapidly-exploring Random Tree)** и **RRT*** в симуляторе MuJoCo.


## Демонстрация работы
| Робот | Демонстрация | Видео на YouTube |
|-------|--------------|----------------|
| **KUKA iiwa14** | <video src="https://github.com/qrvmil/Manipulator/blob/main/pipeline/examples/kuka-iiwa14/kuka_demo_quick.mp4" controls></video> | [Смотреть](https://youtu.be/U0jLNqwuZgs) |
| **Universal Robots UR5e** | <video src="https://github.com/qrvmil/Manipulator/blob/main/pipeline/examples/ur5e/ur5e_demo_quick.mp4" controls></video> | [Смотреть](https://youtu.be/xK5pJgHGtVE) |
| **Franka FR3** | <video src="https://github.com/qrvmil/Manipulator/blob/main/pipeline/examples/fr3/fr3_demo_quick.mp4" controls></video> | [Смотреть](https://youtu.be/FXCSatCztHg) |


Для запуска демонстрации выполните следующие команды:

```bash
# Перейдите в директорию с примерами
cd pipeline/examples

# Для запуска демонстрации KUKA iiwa14
cd kuka-iiwa14
mjpython kuka-iiwa14_demo.py

# Для запуска демонстрации UR5e
cd ../ur5e
mjpython ur5e_demo.py

# Для запуска демонстрации Franka FR3
cd ../fr3
mjpython fr3_demo.py
```

Каждый демонстрационный скрипт содержит предварительно настроенные параметры для соответствующего робота и показывает пример планирования и выполнения движения.

## Добавление собственного робота

Вы можете добавить в проект своего робота и протестировать на нем алгоритмы планирования движения. Для этого следуйте инструкции ниже:

### 1. Подготовка файлов

1. **Создание сцены MuJoCo** (`simulator/models/your_robot/`)
   - Создайте директорию для вашего робота
   - Подготовьте основной файл сцены `scene.xml`
   - Добавьте необходимые 3D модели в `assets/`
   - При необходимости добавьте объекты окружения в `objects/`

2. **Конфигурация робота** (`pipeline/config/robots/your_robot.yaml`)
   - Создайте YAML-файл с настройками робота
   - Укажите количество степеней свободы, ограничения суставов
   - Определите параметры для обратной кинематики
   - Настройте параметры планирования движения
   
   > ⚠️ Важно: Имя файла конфигурации (`your_robot.yaml`) должно соответствовать значению параметра `robot_name` в вашем скрипте.

### 2. Создание демонстрации

1. **Скрипт запуска** (`pipeline/examples/your_robot/your_robot_demo.py`)
   ```python
   from pipeline import Pipeline
   
   def main():
       # Инициализация (имя должно соответствовать названию .yaml файла)
       pipeline = Pipeline(robot_name="your_robot")
       
       # Задание целевых точек
       pipeline.set_targets([
           [x1, y1, z1],
           [x2, y2, z2]
       ])
       
       # Планирование и выполнение движения
       pipeline.plan_path()
       pipeline.run_simulation()
   
   if __name__ == "__main__":
       main()
   ```

2. **Запуск демонстрации**
   ```bash
   cd pipeline/examples/your_robot
   mjpython your_robot_demo.py
   ```

После выполнения скрипта траектория движения будет сохранена в директории `outputs/` с уникальным именем, включающим временную метку.


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

## Поддерживаемые роботы

- **KUKA iiwa 14** 
- **Universal Robots UR5e**
- **Franka FR3**

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
│   │   ├── ur5e/                # Universal Robots UR5e
│   │   └── fr3/                 # Franka FR3
│   └── ik/                      # Обратная кинематика
├── pipeline/                     # Основной пайплайн
│   ├── config/                  # Конфигурационные файлы
│   │   ├── robots/             # Конфиги роботов
│   │   └── simulation_settings.yaml
│   ├── examples/               # Примеры использования
│   └── utils/                 # Вспомогательные функции
├── outputs/                   # Сохраненные траектории
```

## Примечания

- Все сгенерированные траектории сохраняются в директории `outputs/` с уникальными именами
- Временные файлы (например, `qpath.txt`) автоматически удаляются после выполнения
- Для добавления нового робота необходимо создать соответствующие файлы сцены и конфигурации и разместить их в соответствующих директориях (`simulator/models` и `pipeline/config/robots`)  
- Примеры запуска пайплайна с роботом kuka-iiwa14 можно найти в `pipeline/examples`

---

