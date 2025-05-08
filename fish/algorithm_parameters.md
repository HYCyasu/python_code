# 算法参数

以下列出各路径规划与优化算法在调用时使用的默认参数。

---

## 1. A* 算法 (`algorithm_AStar`)

- **无额外参数**，只需传入环境对象 `env`。

---

## 2. RRT 算法 (`algorithm_RRT`)

- `env`: `GridEnvironment` 实例  
- `n_runs`: 100  

### 内部参数（`rrt_planner`）
- `max_iter`: 5000  
- `max_sample_attempts`: 200  
- 步长 `L` (`get_neighbors_rrt`): 1.2  

---

## 3. 遗传算法 (`algorithm_GA`)

- `env`: `GridEnvironment` 实例  
- `population_size`: 100  
- `generations`: 100  
- `crossover_rate`: 0.82  
- `mutation_rate`: 0.063  
- `elite_rate`: 0.03  
- `population_generator`: `None`  
- `AFSAmode`: 0  

---

## 4. AFSA 初始化种群 (`generate_initial_population_afsa`)

- `env`: `GridEnvironment` 实例  
- `population_size`: 与调用时一致  
- `AFSAmode`: 0  
- `max_trials`: 100  

---

## 5. 人工鱼群算法 (`algorithm_AFSA`)

- `env`: `GridEnvironment` 实例  
- `population_size`: 20  
- `iterations`: 100  
- `try_number`: 5  
- `default_delta`: 0.8  
- `default_step`: √2 ≈ 1.4142  
- `default_visual`: 4  
- `default_random_prob`: 0.1  
- `use_variable_step`: True  
- `use_variable_visual`: True  
- `use_variable_crowding`: True  
- `use_variable_random`: True  

---

## 6. 基于 AFSA 初始化的 GA 变体

> 在 `algorithm_GA` 中使用 `generate_initial_population_afsa` 生成初始种群，并通过不同 `AFSAmode` 参数区分：

| 算法名称 | AFSAmode |
|----------|----------|
| GFSA     | 0        |
| GFSA1    | 1        |
| GFSA2    | 2        |
| GFSA3    | 3        |

**共有参数：**

- `env`: `GridEnvironment` 实例  
- `population_size`: 100  
- `generations`: 100  
- `crossover_rate`: 0.82  
- `mutation_rate`: 0.063  
- `elite_rate`: 0.03  
