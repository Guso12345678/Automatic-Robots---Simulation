# AMR Simulation — Robot Autónomo con ROS 2 y CoppeliaSim
 
Proyecto final de la asignatura **Robótica y Sistemas Autónomos (RMA)** (IMAT 3º — Comillas ICAI).  
Simulación completa de un robot móvil autónomo (TurtleBot3 Burger) en CoppeliaSim con ROS 2. Implementa los cuatro pilares de la robótica autónoma: **localización** (Filtro de Partículas + EKF), **planificación de trayectorias** (PRM + A*), **control** (Pure Pursuit + Wall Follower) y **gestión del ciclo de vida** de nodos ROS 2.
 
---
 
## Descripción
 
El sistema navega de forma autónoma en entornos de interior desconocidos. Construye un roadmap probabilístico del entorno, planifica una ruta óptima mediante A*, estima su posición en tiempo real con un filtro de partículas y se desplaza siguiendo la trayectoria calculada con Pure Pursuit. Todo orquestado sobre la arquitectura de nodos y tópicos de **ROS 2**.
 
---
 
## Estructura del proyecto
 
```
src_simulation/
├── amr_bringup/          # Launch files y lifecycle manager
│   └── launch/
│       ├── lab02.launch.py
│       ├── lab03.launch.py
│       ├── lab04.launch.py
│       └── project.launch.py
├── amr_control/          # Controladores de movimiento
│   └── amr_control/
│       ├── pure_pursuit.py          # Controlador Pure Pursuit
│       ├── pure_pursuit_node.py     # Nodo ROS 2
│       ├── wall_follower.py         # Controlador Wall Follower (PD)
│       └── wall_follower_node.py
├── amr_localization/     # Localización del robot
│   └── amr_localization/
│       ├── particle_filter.py       # Filtro de partículas con DBSCAN
│       ├── particle_filter_node.py
│       ├── ekf_filter.py            # Extended Kalman Filter
│       ├── maps.py                  # Gestión del mapa
│       └── intersect.c / .py        # Intersección rayo-pared (C compilado)
├── amr_planning/         # Planificación de trayectorias
│   └── amr_planning/
│       ├── prm.py                   # Probabilistic Roadmap + A* + suavizado
│       ├── prm_node.py
│       └── maps.py
├── amr_simulation/       # Interfaz con CoppeliaSim
│   └── amr_simulation/
│       ├── coppeliasim.py
│       ├── coppeliasim_node.py
│       ├── robot.py
│       └── robot_turtlebot3_burger.py
└── amr_msgs/             # Mensajes ROS 2 personalizados
    └── msg/
        ├── Move.msg
        └── PoseStamped.msg
```
 
---
 
## Cómo ejecutar
 
**Requisitos:** ROS 2 (Humble o superior), CoppeliaSim, Python 3.10+
 
```bash
# Compilar el workspace
colcon build --symlink-install
source install/setup.bash
 
# Lanzar el proyecto completo
ros2 launch amr_bringup project.launch.py
```
 
---
 
## Algoritmos implementados
 
### 1. Planificación de trayectorias (`amr_planning/prm.py`)
 
**PRM — Probabilistic Roadmap** para construcción del grafo de navegación:
 
- Generación aleatoria de nodos en el espacio libre (o en grid uniforme)
- Conexión de nodos a distancia ≤ `connection_distance` sin cruzar obstáculos
- Distancia de seguridad a obstáculos configurable (`obstacle_safety_distance`)
**A\*** sobre el PRM para encontrar el camino óptimo:
 
```python
# f(n) = g(n) + h(n)
tentative_f = tentative_g + np.linalg.norm(neighbor - goal)
```
 
**Suavizado de trayectoria** por optimización iterativa con pesos de datos y suavizado:
 
```python
smoothed[i] += data_weight * (original[i] - smoothed[i]) +
               smooth_weight * (smoothed[i-1] + smoothed[i+1] - 2*smoothed[i])
```
 
---
 
### 2. Localización (`amr_localization/`)
 
**Filtro de Partículas** con ciclo `predict → sense → resample`:
 
- Dispersión inicial global o localización por pose estimada
- Ponderación de partículas basada en comparación de lecturas LiDAR con el mapa
- Clustering con **DBSCAN** para detectar convergencia y reducir el número de partículas automáticamente a 100 cuando hay un solo cluster
- Detección de colisiones rayo-pared en **C compilado** (`intersect.c` / `libintersect.so`) para máxima velocidad
**Extended Kalman Filter (EKF)** para tracking continuo de pose:
 
```python
# Predicción con modelo cinemático diferencial
x_new = x + (v/w) * (sin(θ + w·dt) - sin(θ))
y_new = y + (v/w) * (cos(θ) - cos(θ + w·dt))
 
# Ganancia de Kalman
K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
 
# Actualización
x = x + K·(z - H·x)
P = (I - K·H)·P
```
 
---
 
### 3. Control de movimiento (`amr_control/`)
 
**Pure Pursuit** — seguimiento de trayectoria con lookahead variable:
 
1. Encuentra el punto más cercano del path al robot
2. Busca el punto objetivo a distancia `lookahead_distance` por interpolación
3. Calcula velocidades lineal y angular:
```python
alpha = arctan2(sin(β - θ), cos(β - θ))   # Error de ángulo
v = 0.1                                      # Velocidad lineal constante
w = 2·v·sin(α) / dist_lookahead             # Velocidad angular
```
 
Umbral de rotación pura (`π/8 rad`) para evitar derive al doblar.
 
**Wall Follower** — navegación reactiva con controlador **PD**:
 
```python
error = left_distance - right_distance
w = Kp·error + Kd·(error - prev_error)/dt   # Kp=0.6, Kd=0.07
```
 
Máquina de estados: `FOLLOW_WALL → DETECT_TURN → TURNING → FOLLOW_WALL`
 
---
 
### 4. Gestión de nodos (`amr_bringup/`)
 
`lifecycle_manager_node.py` orquesta el ciclo de vida de todos los nodos ROS 2 (configure → activate → deactivate), garantizando que el sistema arranca y para de forma controlada y determinista.
 
---
 
## Tecnologías
 
- **ROS 2** (Humble) — comunicación entre nodos, tópicos, launch files
- **CoppeliaSim** — simulación física del TurtleBot3 Burger
- Python 3.10 + numpy, matplotlib, sklearn (DBSCAN)
- **C** — biblioteca compilada para intersecciones rayo-pared (`.so` / `.dll` / `.dylib`)
- CMake — compilación de la biblioteca C multiplataforma
---
 
## Autores
 
- **Luis Gervás**
- **Carlos González**
- **Guzman Pérez Ibarz**
Proyecto de laboratorio — Robótica y Sistemas Autónomos (RMA), 3º Ingeniería Matemática e Inteligencia Artificial (IMAT), Comillas ICAI. Curso 2024/25.
