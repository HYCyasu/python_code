import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import math
import matplotlib

# 配置 matplotlib 以支持中文显示和正确渲染负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，确保中文正常显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
history_AStar = []
history_RRT = []
history_AFSA = []
history_GA = []
history_GFSA = []
history_GFSA1 = []
history_GFSA2 = []
history_GFSA3 = []

# 定义八个可移动方向：上下左右和四个对角线，表示为(行偏移, 列偏移)
MOVE_DIRECTIONS = [
    (0, 1),  # 向右移动一格
    (1, 0),  # 向下移动一格
    (0, -1),  # 向左移动一格
    (-1, 0),  # 向上移动一格
    (1, 1),  # 向右下对角移动一格
    (1, -1),  # 向左下对角移动一格
    (-1, 1),  # 向右上对角移动一格
    (-1, -1)  # 向左上对角移动一格
]


def generate_fixed_grid():
    """
    创建并返回一个预设障碍的 20×20 网格地图

    返回值:
      - grid: numpy.ndarray, 形状(20,20), 值为0(可通行)或1(障碍)
      - start: tuple(int, int), 起点坐标 (0, 0)
      - goal: tuple(int, int), 终点坐标 (19, 19)

    障碍物布局说明:
      障碍坐标列表中列出了多个 (row, col) 对，每个对应网格中不可通行的位置。
      通过循环将这些坐标在 grid 中标记为 1。

    使用示例:0
      grid, start, goal = generate_fixed_grid()
    """
    rows, cols = 20, 20  # 地图行数和列数
    grid = np.zeros((rows, cols), dtype=int)  # 初始化全零网格

    # 手动定义障碍物位置列表，每个元素为 (row, col)
    obstacles = [
        (0, 10),
        (1, 5), (1, 6), (1, 7), (1, 12),
        (2, 3), (2, 4), (2, 5), (2, 10), (2, 13),
        (3, 1), (3, 8), (3, 12), (3, 14),
        (4, 7), (4, 8), (4, 9), (4, 15),
        (5, 0), (5, 3), (5, 11), (5, 12),
        (6, 4), (6, 5), (6, 10), (6, 13),
        (7, 2), (7, 3), (7, 7), (7, 8), (7, 16),
        (8, 6), (8, 7), (8, 14), (8, 15),
        (9, 1), (9, 2), (9, 10), (9, 11),
        (10, 5), (10, 8), (10, 12), (10, 17),
        (11, 3), (11, 4), (11, 9), (11, 10),
        (12, 1), (12, 7), (12, 8), (12, 15), (12, 18),
        (13, 0), (13, 2), (13, 6), (13, 11),
        (14, 4), (14, 5), (14, 14),
        (15, 3), (15, 9), (15, 12), (15, 16),
        (16, 6), (16, 7), (16, 8), (16, 17),
        (17, 1), (17, 10), (17, 11), (17, 15),
        (18, 5), (18, 12), (18, 14),
        (19, 0), (19, 9)
    ]

    # 在 grid 中将障碍点标记为 1
    for r, c in obstacles:
        grid[r, c] = 1

    # 定义起点和终点
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    return grid, start, goal


class GridEnvironment:
    """
    栅格环境类: 封装地图数据并提供邻居查询功能

    属性:
      - grid: numpy.ndarray, 网格地图
      - start: tuple, 起点坐标
      - goal: tuple, 终点坐标
      - rows: int, 地图行数
      - cols: int, 地图列数
      - free_cells: set of tuple, 所有可通行格子集合

    方法:
      - get_neighbors(cell): 获取给定格子的所有合法八方向邻居
    """

    def __init__(self, grid, start, goal):
        self.grid = grid  # 存储地图数据
        self.start = start  # 存储起点
        self.goal = goal  # 存储终点
        self.rows, self.cols = grid.shape  # 地图尺寸
        # 构建 free_cells 集合，包含所有 grid == 0 的坐标
        self.free_cells = {
            (i, j)
            for i in range(self.rows)
            for j in range(self.cols)
            if grid[i, j] == 0
        }

    def get_neighbors(self, cell):
        """
        获取指定格子的所有合法邻居

        输入:
          - cell: tuple(int, int), 当前格子 (row, col)

        输出:
          - neighbors: list of tuple, 满足以下条件的邻居坐标列表:
            1. 在地图边界内
            2. 不是障碍 (grid value == 0)
            3. 对角移动时，确保拐角处的水平与垂直方向都可通行，防止穿越障碍角落
        """
        i, j = cell
        neighbors = []
        for d in MOVE_DIRECTIONS:

            ni, nj = i + d[0], j + d[1]
            # 越界检查
            if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols:
                continue
            # 障碍物检查
            if self.grid[ni, nj] == 1:
                continue
            # 对角移动拐角检查
            if d[0] != 0 and d[1] != 0:
                if self.grid[i + d[0], j] == 1 or self.grid[i, j + d[1]] == 1:
                    continue
            neighbors.append((ni, nj))
        return neighbors


def generate_random_path_from(env, start, goal, max_steps=200):
    """
    随机生成一条从起点到终点的路径，用于基准测试或算法初始化

    输入:
      - env: GridEnvironment 实例
      - start: tuple, 起点坐标
      - goal: tuple, 终点坐标
      - max_steps: int, 最大步数限制，防止无解时死循环

    输出:
      - path: list of tuple, 成功时返回坐标列表；失败时返回 None

    算法逻辑:
      1. 初始化 path 和 visited 集合，当前节点设为 start
      2. 重复直到到达 goal 或步数超限:
         a. 获取当前节点的所有未访问邻居
         b. 若无可选邻居, 返回 None
         c. 随机选择一个邻居作为下一个节点, 更新 path 和 visited
      3. 如果成功到达 goal, 返回 path; 否则返回 None
    """
    path = [start]
    visited = {start}
    current = start
    steps = 0

    while current != goal and steps < max_steps:
        nbrs = env.get_neighbors(current)
        # 排除已访问格子
        nbrs = [n for n in nbrs if n not in visited]
        if not nbrs:
            return None
        next_node = random.choice(nbrs)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
        steps += 1

    return path if current == goal else None


def move_cost(d):
    """
    计算一个移动方向的代价

    输入:
      - d: tuple(int, int), 方向向量 (delta_row, delta_col)
    输出:
      - float, 对角移动返回 sqrt(2)，水平/垂直移动返回 1
    公式:
      cost = sqrt(d[0]**2 + d[1]**2)
    """
    return math.hypot(d[0], d[1])


def path_cost(path):
    """
    计算一条路径的总代价

    输入:
      - path: list of tuple, 坐标列表
    输出:
      - float, 累计的 move_cost 代价
      - 如果 path 为 None, 返回 float('inf') 表示不可达

    计算方法:
      遍历相邻两点, 通过 move_cost 计算每一步代价并累加
    """
    if path is None:
        return float('inf')
    cost = 0.0
    for prev, curr in zip(path[:-1], path[1:]):
        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]
        cost += move_cost((dr, dc))
    return cost


# -------------------------------
# 2. 算法模块接口
# -------------------------------

# （2.1）A* 算法模块
def heuristic(a, b):
    """
    八向距离启发函数，兼容统一移动规则
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def reconstruct_path(came_from, start, goal):
    """从 came_from 字典重构从 start 到 goal 的路径"""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path


def astar_planner(env, iterations=100, w_init=50.0):
    """
    Anytime Weighted A*：
      - 从权重 w_init 线性下降到 1.0，共 iterations 步
      - 每一步运行一次带权 A*，找到可行解后更新全局最优并记录到 history_AStar

    返回:
      best_path, best_cost
    """
    global history_AStar
    history_AStar.clear()

    weights = np.linspace(w_init, 5, iterations)

    grid, start, goal = env.grid, env.start, env.goal
    best_path = None
    best_cost = 50

    for w in weights:
        # 带权 A* 初始化
        open_set = [(w * heuristic(start, goal), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            f_curr, g_curr, current = heapq.heappop(open_set)

            if current == goal:
                # 重构路径并评估
                path = reconstruct_path(came_from, start, goal)
                cost_here = path_cost(path)
                if cost_here < best_cost:
                    best_cost = cost_here
                    best_path = path
                break

            # 扩展邻居
            for d in MOVE_DIRECTIONS:
                nbr = (current[0] + d[0], current[1] + d[1])
                # 越界检查
                if not (0 <= nbr[0] < env.rows and 0 <= nbr[1] < env.cols):
                    continue
                # 障碍检查
                if grid[nbr] == 1:
                    continue
                # 对角拐角检查
                if d[0] != 0 and d[1] != 0:
                    if grid[current[0] + d[0], current[1]] == 1 or grid[current[0], current[1] + d[1]] == 1:
                        continue

                tentative_g = g_curr + move_cost(d)
                if nbr not in g_score or tentative_g < g_score[nbr]:
                    g_score[nbr] = tentative_g
                    came_from[nbr] = current
                    f_nbr = tentative_g + w * heuristic(nbr, goal)
                    heapq.heappush(open_set, (f_nbr, tentative_g, nbr))

        # 记录当前权重下的全局最优代价
        history_AStar.append(best_cost)

    return best_path, best_cost


def algorithm_AStar(env):
    print("运行 A* 算法...")
    best_path = None
    best_cost = float('inf')
    # 调用 A* 得到路径和代价
    path, cost = astar_planner(env)
    if cost < best_cost:
        best_cost = cost
        best_path = path
    print("A* 最终记录路径代价:", best_cost)
    return best_path, best_cost


# （2.2）RRT 算法模块
def distance(a, b):
    """
    欧氏距离计算
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors_rrt(node, target, L=0.8):
    """
    根据当前节点 node 和采样目标 target 生成新节点，步长约为 L
    """
    dx = target[0] - node[0]
    dy = target[1] - node[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return node
    new_x = int(round(node[0] + (dx / dist) * L))
    new_y = int(round(node[1] + (dy / dist) * L))
    new_node = (new_x, new_y)
    if new_node == node:
        new_node = (node[0] + int(math.copysign(1, dx)), node[1] + int(math.copysign(1, dy)))
    return new_node


def rrt_planner(env, max_iter=5000, max_sample_attempts=200):
    """
    RRT 算法接口：在环境中构建搜索树，当有节点距离目标足够近时回溯生成路径
    """
    grid = env.grid
    start = env.start
    goal = env.goal
    tree = {start: None}
    nodes = [start]

    for _ in range(max_iter):
        for _ in range(max_sample_attempts):
            rand = (random.randint(0, env.rows - 1), random.randint(0, env.cols - 1))
            if grid[rand] == 0:
                break
        else:
            continue

        nearest = min(nodes, key=lambda n: distance(n, rand))
        new_node = get_neighbors_rrt(nearest, rand)
        if not (0 <= new_node[0] < env.rows and 0 <= new_node[1] < env.cols):
            continue
        if grid[new_node] == 1 or new_node in tree:
            continue

        # 若是对角移动，检查对应正交方向
        dx = new_node[0] - nearest[0]
        dy = new_node[1] - nearest[1]
        if dx != 0 and dy != 0:
            neighbor1 = (nearest[0] + dx, nearest[1])
            neighbor2 = (nearest[0], nearest[1] + dy)
            if not (0 <= neighbor1[0] < env.rows and 0 <= neighbor1[1] < env.cols) or grid[neighbor1] == 1:
                continue
            if not (0 <= neighbor2[0] < env.rows and 0 <= neighbor2[1] < env.cols) or grid[neighbor2] == 1:
                continue

        tree[new_node] = nearest
        nodes.append(new_node)

        if distance(new_node, goal) <= 1:
            tree[goal] = new_node if new_node != goal else nearest
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = tree[node]
            return path[::-1], path_cost(path)
    return None, float('inf')


def algorithm_RRT(env, n_runs=100):
    """
    多次运行 RRT 算法，迭代执行 n_runs 次并记录每次迭代中当前最优代价形成收敛曲线。

    参数：
    - env: 环境实例；
    - n_runs: 迭代次数，默认为 100 次。
    """
    global history_RRT
    history_RRT.clear()
    print("运行 RRT 算法...")
    best_path = None
    best_cost = 50
    for i in range(n_runs):
        path, cost = rrt_planner(env)
        history_RRT.append(best_cost)
        # 更新当前最佳代价和路径：仅当产生了有效路径且其代价更优时更新
        if path is not None and cost < best_cost:
            best_cost = cost
            best_path = path

        # 若尚未找到有效路径，则记录 None；或者使用 best_cost 来记录当前最优状态
    print("RRT 最终路径代价:", best_cost)
    return best_path, best_cost


# （2.3）遗传算法模块（GA）
def GA(env, population_size=50, generations=100,
       crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03,
       population_generator=None, AFSAmode=0):
    """
    遗传算法模块：搜索从起点到终点的路径，返回最优路径及其代价
    参数：
      population_generator: 可选的种群生成函数，输入 (env, population_size)；若为空则使用随机生成方法
    """
    history = []
    if population_generator is None:
        # 默认使用随机生成种群
        population = []
        while len(population) < population_size:
            p = generate_random_path_from(env, env.start, env.goal)
            if p is not None:
                population.append(p)
    else:
        # 使用指定方法生成初始种群
        # 注意这里参数顺序已经调整成 (env, population_size, AFSAmode, max_trials)
        population = population_generator(env, population_size, AFSAmode=AFSAmode, max_trials=None)

    best_solution = None
    best_cost = 50

    def tournament_selection(pop, fitnesses, tournament_size=2):
        selected = []
        for _ in range(len(pop)):
            contenders = random.sample(list(zip(pop, fitnesses)), tournament_size)
            winner = min(contenders, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(parent1, parent2):
        candidates = [i for i in range(len(parent1) - 1) if parent1[i] in parent2]
        if not candidates:
            return parent1, parent2
        i = random.choice(candidates)
        next_index = None
        for j in range(i + 1, len(parent1)):
            if parent1[j] in parent2:
                next_index = j
                break
        if next_index is None:
            return parent1, parent2
        common_A = parent1[i]
        common_B = parent1[next_index]
        idx1_A, idx1_B = i, next_index
        idx2_A = parent2.index(common_A)
        idx2_B = parent2.index(common_B)
        child1 = parent1[:idx1_A + 1] + parent2[idx2_A + 1: idx2_B] + parent1[idx1_B:]
        child2 = parent2[:idx2_A + 1] + parent1[idx1_A + 1: idx1_B] + parent2[idx2_B:]
        return child1, child2

    def mutate(individual, env):
        if len(individual) <= 3:
            return individual
        a, b = sorted(random.sample(range(1, len(individual) - 1), 2))
        new_segment = generate_random_path_from(env, individual[a], individual[b])
        if new_segment is None or len(new_segment) < 2:
            return individual
        return individual[:a + 1] + new_segment[1:-1] + individual[b:]

    for gen in range(generations):
        fitnesses = [path_cost(ind) for ind in population]
        for ind, cost in zip(population, fitnesses):
            if cost < best_cost:
                best_cost = cost
                best_solution = ind
        history.append(best_cost)
        selected = tournament_selection(population, fitnesses, tournament_size=2)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            offspring.extend([c1, c2])
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i], env)
        elite_count = max(1, int(elite_rate * population_size))
        sorted_population = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0])]
        population = sorted_population[:elite_count] + offspring[:population_size - elite_count]
    return best_solution, best_cost, history


# （2.3.1）利用 AFSA 算法生成初始种群的辅助函数
def generate_initial_population_afsa(env, population_size, AFSAmode=0, max_trials=None):
    """
    利用 AFSA 算法多次生成可行路径作为初始种群
    若连续 max_trials 次未能生成足够个体，则对剩余个体采用随机搜索补全。
    """
    if max_trials is None:
        max_trials = 100
    population = []
    trials = 0
    while len(population) < population_size and trials < max_trials:
        if AFSAmode == 1:
            path, cost = algorithm_AFSA_generate(env, use_variable_step=True, use_variable_visual=True,
                                                 use_variable_crowding=False, use_variable_random=False)
        elif AFSAmode == 2:
            path, cost = algorithm_AFSA_generate(env, use_variable_step=False, use_variable_visual=False,
                                                 use_variable_crowding=True, use_variable_random=False)
        elif AFSAmode == 3:
            path, cost = algorithm_AFSA_generate(env, use_variable_step=False, use_variable_visual=False,
                                                 use_variable_crowding=False, use_variable_random=True)
        else:
            path, cost = algorithm_AFSA_generate(env, use_variable_step=True, use_variable_visual=True,
                                                 use_variable_crowding=True, use_variable_random=True)

        if path is not None and cost < float('inf'):
            population.append(path)
        trials += 1
    # 若不足则补充随机路径
    while len(population) < population_size:
        p = generate_random_path_from(env, env.start, env.goal)
        if p is not None:
            population.append(p)
    return population


# （2.4）人工鱼群算法（AFSA）模块 —— 新实现
# 此处不做收敛记录修改，因其主要用于种群生成，实际迭代历史会在 algorithm_AFSA 中记录

def grideAF_foodconsistence(X, goal):
    if isinstance(X, tuple):
        return math.hypot(X[0] - goal[0], X[1] - goal[1])
    else:
        return [math.hypot(pos[0] - goal[0], pos[1] - goal[1]) for pos in X]


def variable_visual_by_iter(cur_iter, max_iter, vis_min=math.sqrt(2), vis_max=4):
    ratio = cur_iter / max_iter
    alpha = math.exp(-20 * (ratio ** 5))
    return alpha * vis_max + (1 - alpha) * vis_min


def crowding_factor_by_distance(pos, goal, default_delta, nf):
    d = distance(pos, goal)
    threshold = 2 * math.sqrt(2)
    if d <= threshold:
        if d == 0:
            return 1.0 / nf
        return math.tanh(1.0 / d) / nf
    else:
        return default_delta


def step_by_iter(cur_iter, max_iter, default_step):
    ratio = cur_iter / max_iter
    alpha = math.exp(-20 * (ratio ** 5))
    return alpha * default_step + 0.3


def prob_by_iter(no_improve, max_iter, default_random_prob):
    ratio = no_improve / max_iter
    return default_random_prob * ratio


def grid_af_prey(n, pos, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter, default_step, use_variable_step):
    present_H = lastH[ii]
    current_step = step_by_iter(cur_iter, MAXGEN, default_step) if use_variable_step else default_step
    allow_area = [p for p in allowed if 0 < distance(pos, p) <= current_step]
    m = random.choices([0, 1], weights=[0.2, 0.8])[0]
    nextPosition = None
    if m == 0:
        for _ in range(try_number):
            if not allow_area:
                break
            candidate = random.choice(allow_area)
            if present_H > grideAF_foodconsistence(candidate, goal):
                nextPosition = candidate
                break
    else:
        H_min = present_H
        for p in allow_area:
            candidate_H = grideAF_foodconsistence(p, goal)
            if candidate_H < H_min:
                H_min = candidate_H
                nextPosition = p
    if nextPosition is None:
        nextPosition = random.choice(allow_area) if allow_area else pos
    return nextPosition, grideAF_foodconsistence(nextPosition, goal)


def grid_af_follow(n, positions, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                   default_visual, use_variable_visual,
                   default_delta, use_variable_crowding,
                   default_step, use_variable_step):
    Xi = positions[ii]
    dist_list = [distance(Xi, other) for other in positions]
    visual = variable_visual_by_iter(cur_iter, MAXGEN) if use_variable_visual else default_visual
    indices = [i for i, d in enumerate(dist_list) if d > 0 and d < visual]
    nf = len(indices)
    allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
    if nf > 0:
        Xvisual = [positions[i] for i in indices]
        Hvisual = [lastH[i] for i in indices]
        Xmin = Xvisual[Hvisual.index(min(Hvisual))]
        Hi = lastH[ii]
        delta_val = crowding_factor_by_distance(Xi, goal, default_delta, nf) if use_variable_crowding else default_delta
        if (min(Hvisual) / nf) <= (Hi * (1 - delta_val)):
            for _ in range(try_number):
                if not allow_area:
                    break
                candidate = random.choice(allow_area)
                if distance(candidate, Xmin) < distance(Xi, Xmin):
                    return candidate, grideAF_foodconsistence(candidate, goal)
            return Xi, grideAF_foodconsistence(Xi, goal)
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                                default_step, use_variable_step)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                            default_step, use_variable_step)


def grid_af_swarm(n, positions, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                  default_visual, use_variable_visual,
                  default_delta, use_variable_crowding,
                  default_step, use_variable_step):
    Xi = positions[ii]
    dist_list = [distance(Xi, other) for other in positions]
    visual = variable_visual_by_iter(cur_iter, MAXGEN) if use_variable_visual else default_visual
    indices = [i for i, d in enumerate(dist_list) if d > 0 and d < visual]
    nf = len(indices)
    allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
    if nf > 0:
        avg_r = math.ceil(sum(positions[i][0] for i in indices) / nf)
        avg_c = math.ceil(sum(positions[i][1] for i in indices) / nf)
        Xc = (avg_r, avg_c)
        Hc = grideAF_foodconsistence(Xc, goal)
        Hi = lastH[ii]
        delta_val = crowding_factor_by_distance(Xi, goal, default_delta, nf) if use_variable_crowding else default_delta
        if (Hc / nf) <= (Hi * (1 - delta_val)):
            for _ in range(try_number):
                if not allow_area:
                    break
                candidate = random.choice(allow_area)
                if distance(candidate, Xc) < distance(Xi, Xc):
                    return candidate, grideAF_foodconsistence(candidate, goal)
            return Xi, grideAF_foodconsistence(Xi, goal)
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                                default_step, use_variable_step)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter,
                            default_step, use_variable_step)


def algorithm_AFSA_generate(env,
                            population_size=3,
                            iterations=60,
                            try_number=3,
                            default_delta=0.8,
                            default_step=math.sqrt(1),
                            default_visual=3,
                            default_random_prob=0.1,
                            use_variable_step=False,
                            use_variable_visual=False,
                            use_variable_crowding=False,
                            use_variable_random=False):
    allowed = sorted(env.free_cells)
    N = population_size
    MAXGEN = iterations

    positions = [env.start for _ in range(N)]
    trajectories = [[env.start] for _ in range(N)]
    H = [grideAF_foodconsistence(env.start, env.goal) for _ in range(N)]

    global_best_cost = 50
    global_best_path = None
    no_update_rounds = 0

    for j in range(MAXGEN):
        old_global_best = global_best_cost
        new_positions = []
        for i in range(N):
            Xi = positions[i]
            current_cost = grideAF_foodconsistence(Xi, env.goal)
            pos_swarm, cost_swarm = grid_af_swarm(
                N, positions, i, try_number, H, allowed, env.goal, MAXGEN, j,
                default_visual, use_variable_visual,
                default_delta, use_variable_crowding,
                default_step, use_variable_step
            )
            pos_follow, cost_follow = grid_af_follow(
                N, positions, i, try_number, H, allowed, env.goal, MAXGEN, j,
                default_visual, use_variable_visual,
                default_delta, use_variable_crowding,
                default_step, use_variable_step
            )
            if cost_swarm < current_cost or cost_follow < current_cost:
                candidate, candidate_cost = (pos_follow, cost_follow) if cost_follow < cost_swarm else (
                    pos_swarm, cost_swarm)
            else:
                candidate, candidate_cost = grid_af_prey(
                    N, Xi, i, try_number, H, allowed, env.goal, MAXGEN, j,
                    default_step, use_variable_step
                )
                if candidate == Xi:
                    current_rand_prob = (prob_by_iter(no_update_rounds, MAXGEN, default_random_prob)
                                         if use_variable_random else default_random_prob)
                    if random.random() < current_rand_prob:
                        allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
                        if allow_area:
                            candidate = random.choice(allow_area)
                            candidate_cost = grideAF_foodconsistence(candidate, env.goal)
            new_positions.append(candidate)
            H[i] = candidate_cost
            trajectories[i].append(candidate)
        positions = new_positions.copy()

        reached = [i for i, pos in enumerate(positions) if pos == env.goal]
        if reached:
            for idx in reached:
                fish_path_cost = path_cost(trajectories[idx])
                if fish_path_cost < global_best_cost:
                    global_best_cost = fish_path_cost
                    global_best_path = trajectories[idx]
            if global_best_cost == old_global_best:
                no_update_rounds += 1
            else:
                no_update_rounds = 0
            break

    if global_best_path is None:
        return None, float('inf')
    else:
        return global_best_path, global_best_cost


def algorithm_AFSA(env, generations=100):
    """
    AFSA 主入口：调用 algorithm_AFSA_generate 进行多代迭代，
    并借助全局 history_AFSA 列表记录每代最优总代价。

    参数:
      - env: GridEnvironment 对象
      - iterations: 迭代次数

    返回:
      - best_path: 最优路径列表
      - best_cost: 最优总代价
    """
    best_path = []
    best_cost = 50
    # 调用你已实现的 AFSA 生成函数
    for i in range(generations):
        history_AFSA.append(best_cost)
        path, cost = algorithm_AFSA_generate(env, population_size=30, default_step=math.sqrt(1.7))
        if cost < best_cost:
            best_cost = cost
            best_path = path
    return best_path, best_cost


def algorithm_GA(env, population_size=100, generations=100,
                 crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    global history_GA
    history_GA.clear()
    print("运行GA 算法...")
    path, cost, history = GA(env, population_size, generations,
                             crossover_rate, mutation_rate, elite_rate,
                             population_generator=None, AFSAmode=0)
    history_GA[:] = history
    return path, cost


# （2.5）新算法：利用 AFSA 生成初始种群，再由 GA 搜索
def algorithm_GFSA(env, population_size=300, generations=100,
                   crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    """
    新算法：使用 AFSA 生成初始种群，再利用 GA 进行优化
    直接调用 GA，并将 population_generator 参数设为 generate_initial_population_afsa
    """
    global history_GFSA
    history_GFSA.clear()
    print("运行基于 AFSA 初始化的 GA 算法...")
    path, cost, history = GA(env, population_size, generations,
                             crossover_rate, mutation_rate, elite_rate,
                             population_generator=generate_initial_population_afsa, AFSAmode=0)
    history_GFSA[:] = history
    return path, cost


def algorithm_GFSA1(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    global history_GFSA1
    history_GFSA1.clear()
    print("运行基于 AFSA1 初始化的 GA 算法...")
    path, cost, history = GA(env, population_size, generations,
                             crossover_rate, mutation_rate, elite_rate,
                             population_generator=generate_initial_population_afsa, AFSAmode=1)
    history_GFSA1[:] = history
    return path, cost


def algorithm_GFSA2(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    global history_GFSA2
    history_GFSA2.clear()
    print("运行基于 AFSA2 初始化的 GA 算法...")
    path, cost, history = GA(env, population_size, generations,
                             crossover_rate, mutation_rate, elite_rate,
                             population_generator=generate_initial_population_afsa, AFSAmode=2)
    history_GFSA2[:] = history
    return path, cost


def algorithm_GFSA3(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    global history_GFSA3
    history_GFSA3.clear()
    print("运行基于 AFSA3 初始化的 GA 算法...")
    path, cost, history = GA(env, population_size, generations,
                             crossover_rate, mutation_rate, elite_rate,
                             population_generator=generate_initial_population_afsa, AFSAmode=3)
    history_GFSA3[:] = history
    return path, cost


# -------------------------------
# 3. 绘图辅助函数
# -------------------------------
def plot_paths(env, results, title="路径规划算法对比"):
    """
    绘制环境及各算法路径对比，并为每条路径指定不同的颜色、线型和 Marker
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='Greys', origin='upper')
    plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(0.5, env.cols, 1), [])
    plt.yticks(np.arange(0.5, env.rows, 1), [])
    plt.title(title)
    plt.plot(env.start[1], env.start[0], marker='o', color='g', markersize=12, label="起点")  # 绿色圆形
    plt.plot(env.goal[1], env.goal[0], marker='D', color='m', markersize=12, label="终点")  # 洋红菱形

    colors = ['r', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple']
    line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']

    for idx, (name, path) in enumerate(results.items()):
        if not path:
            continue
        color = colors[idx % len(colors)]
        ls = line_styles[idx % len(line_styles)]
        mkr = markers[idx % len(markers)]
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys,
                 color=color,
                 linestyle=ls,
                 marker=mkr,
                 markersize=6,
                 linewidth=2,
                 label=name)

    plt.legend(loc="upper right", frameon=True)
    plt.show()


def plot_convergence(convergence_histories, title="算法收敛曲线比较", markers=10):
    """
    绘制各算法收敛曲线，并通过 markevery 控制每条曲线的 marker 数量。

    参数:
      convergence_histories: dict, {算法名: 收敛代价列表}
      title: 图表标题
      markers: 每条曲线上大约绘制的 marker 数量
    """
    plt.figure(figsize=(8, 6))

    colors = ['r', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple']
    line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    markers_list = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']

    for idx, (name, hist) in enumerate(convergence_histories.items()):
        if not hist:
            continue
        color = colors[idx % len(colors)]
        ls = line_styles[idx % len(line_styles)]
        mkr = markers_list[idx % len(markers_list)]
        # 根据序列长度和期望 marker 数量计算 markevery
        N = len(hist)
        interval = max(1, N // markers)
        plt.plot(
            hist,
            label=name,
            color=color,
            linestyle=ls,
            marker=mkr,
            markersize=5,
            linewidth=1.5,
            markevery=interval
        )

    plt.xlabel("迭代次数/采样次数")
    plt.ylabel("最优总代价")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="best", frameon=True)
    plt.show()


# -------------------------------
# 4. 主函数
# -------------------------------
def main():
    grid, start, goal = generate_fixed_grid()
    env = GridEnvironment(grid, start, goal)
    algorithms = {
        "AStar": algorithm_AStar,
        "RRT": algorithm_RRT,
        "AFSA": algorithm_AFSA,
        # 默认 GA 使用随机生成初始种群
        "GA_Random": algorithm_GA,
        # 新算法：利用 AFSA 初始化种群的 GA 版本
        "GFSA": algorithm_GFSA,
        # # 可根据需要启用其他 GFSA 变体
        "GFSA1": algorithm_GFSA1,
        "GFSA2": algorithm_GFSA2,
        "GFSA3": algorithm_GFSA3,
    }
    convergence_histories = {
        "RRT": history_RRT,
        "AFSA": history_AFSA,
        "GA_Random": history_GA,
        "GFSA": history_GFSA,
        "GFSA1": history_GFSA1,
        "GFSA2": history_GFSA2,

        "GFSA3": history_GFSA3,
    }
    results = {}
    for name, algo in algorithms.items():
        print(f"\n运行算法：{name}")
        path, cost = algo(env)
        print(f"{name} 返回：代价 = {cost}, 路径 = {path}")
        results[name] = path
    plot_paths(env, results, title="统一代价函数与移动方式下的路径规划对比")
    plot_convergence(convergence_histories, title="各算法收敛曲线对比")


if __name__ == "__main__":
    main()
