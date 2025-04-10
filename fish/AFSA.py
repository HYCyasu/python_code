import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import math
import matplotlib

# -------------------------------
# 全局配置：支持中文及样式设置
# -------------------------------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# -------------------------------
# 1. 环境与公共函数模块
# -------------------------------

def generate_fixed_grid():
    """
    生成固定地图（15×15）和障碍布局
    """
    rows, cols = 15, 15
    grid = np.zeros((rows, cols), dtype=int)
    obstacles = [
        (0, 7), (1, 7), (1, 13), (2, 9), (2, 10), (1, 9), (2, 13), (3, 3), (3, 4),
        (3, 7), (3, 8), (3, 13), (4, 8), (5, 1), (5, 7), (5, 8), (5, 10), (5, 12),
        (6, 3), (6, 6),(6, 7), (7, 3), (7, 4), (7, 6), (7, 7), (7, 12), (7, 14), (8, 8),
        (8, 11), (8, 12), (9, 2), (10, 2), (10, 4), (10, 13), (11, 8),
        (11, 9), (11, 10), (12, 3), (12, 4), (12, 6), (12, 10), (13, 0),
        (13, 4), (13, 6)
    ]
    for r, c in obstacles:
        grid[r, c] = 1
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    return grid, start, goal


class GridEnvironment:
    """
    环境类，封装地图、起点和终点，同时预计算所有可通行的位置
    """

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape
        # 预计算非障碍格子（便于查询 allowed_positions）
        self.free_cells = {(i, j) for i in range(self.rows) for j in range(self.cols) if grid[i, j] == 0}

    def get_neighbors(self, cell):
        """
        返回 cell 的所有合法邻居（依据统一移动规则），检查边界及障碍；
        对于对角移动，需保证水平和竖直方向邻居均可通行（防止拐角穿越）
        """
        i, j = cell
        neighbors = []
        for d in MOVE_DIRECTIONS:
            ni, nj = i + d[0], j + d[1]
            if not (0 <= ni < self.rows and 0 <= nj < self.cols):
                continue
            if self.grid[ni, nj] == 1:
                continue
            if d[0] != 0 and d[1] != 0:
                n1 = (i + d[0], j)
                n2 = (i, j + d[1])
                if not (0 <= n1[0] < self.rows and 0 <= n1[1] < self.cols and self.grid[n1] == 0):
                    continue
                if not (0 <= n2[0] < self.rows and 0 <= n2[1] < self.cols and self.grid[n2] == 0):
                    continue
            neighbors.append((ni, nj))
        return neighbors


# 定义允许的移动方向（八个方向）
MOVE_DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]


def generate_random_path_from(env, start, goal, max_steps=150):
    """
    随机生成一条从起点到终点的路径
    """
    path = [start]
    visited = {start}
    current = start

    while current != goal and len(path) < max_steps:
        nbrs = env.get_neighbors(current)
        nbrs = [n for n in nbrs if n not in visited]
        if not nbrs:
            return None
        next_node = random.choice(nbrs)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path if current == goal else None


def move_cost(d):
    """
    根据移动方向 d 返回代价：对角移动为 sqrt(2)，水平和竖直移动代价为 1
    """
    return math.hypot(d[0], d[1])


def path_cost(path):
    """
    计算路径中连续点之间的欧氏距离之和
    """
    if path is None:
        return float('inf')
    cost = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        cost += move_cost((dx, dy))
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


def astar_planner(env):
    """
    A* 算法接口，返回：路径列表和总代价
    """
    grid = env.grid
    start, goal = env.start, env.goal
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
            return path, path_cost(path)

        for d in MOVE_DIRECTIONS:
            neighbor = (current[0] + d[0], current[1] + d[1])
            # 边界及障碍检查
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            if grid[neighbor] == 1:
                continue
            # 对角移动防止拐角穿越
            if d[0] != 0 and d[1] != 0:
                n1 = (current[0] + d[0], current[1])
                n2 = (current[0], current[1] + d[1])
                if not (0 <= n1[0] < grid.shape[0] and 0 <= n1[1] < grid.shape[1] and grid[n1] == 0):
                    continue
                if not (0 <= n2[0] < grid.shape[0] and 0 <= n2[1] < grid.shape[1] and grid[n2] == 0):
                    continue

            tentative_g = g_score[current] + move_cost(d)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + 3 * heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return None, float('inf')


def algorithm_AStar(env):
    """
    统一接口：调用 A* 算法模块
    """
    print("运行 A* 算法...")
    path, cost = astar_planner(env)
    print("A* 路径代价:", cost)
    return path, cost


# （2.2）RRT 算法模块
def distance(a, b):
    """
    欧氏距离计算
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors_rrt(node, target, L=1.2):
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


def rrt_planner(env, max_iter=5000, max_sample_attempts=100):
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
            return path[::-1]
    return None


def algorithm_RRT(env, n_runs=100):
    """
    多次运行 RRT，选择总代价最低的路径作为最终解
    """
    best_path = None
    best_cost = float('inf')
    for _ in range(n_runs):
        path = rrt_planner(env)
        if path is not None:
            cost = path_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path
    return best_path, best_cost


# （2.3）遗传算法模块（GA）
# 在原有算法基础上，增加可选参数 population_generator，
# 若为空则用随机路径生成种群，否则调用指定的种群生成函数。
def algorithm_GA(env, population_size=100, generations=100,
                 crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03,
                 population_generator=None):
    """
    遗传算法模块：搜索从起点到终点的路径，返回最优路径及其代价
    参数：
      population_generator: 可选的种群生成函数，输入 (env, population_size)；若为空则使用随机生成方法
    """
    if population_generator is None:
        # 默认使用随机生成种群
        population = []
        while len(population) < population_size:
            p = generate_random_path_from(env, env.start, env.goal)
            if p is not None:
                population.append(p)
    else:
        # 采用指定方法生成初始种群
        population = population_generator(env, population_size)

    best_solution = None
    best_cost = float('inf')

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
    return best_solution, best_cost


# （2.3.1）利用 AFSA 算法生成初始种群的辅助函数
def generate_initial_population_afsa(env, population_size, max_trials=None):
    """
    利用 AFSA 算法多次生成可行路径作为初始种群
    若连续 max_trials 次未能生成足够个体，则对剩余个体采用随机搜索补全。
    """
    if max_trials is None:
        max_trials = population_size * 5
    population = []
    trials = 0
    while len(population) < population_size and trials < max_trials:
        # AFSA 算法内部已采用多个鱼进行搜索，此处调用得到一个解
        path, cost = algorithm_AFSA(env)
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
def grideAF_foodconsistence(X, goal):
    """
    计算“食物浓度”：当前位置到目标的欧氏距离；
    参数 X 可为单个位置（元组）或多个位置（列表）
    """
    if isinstance(X, tuple):
        return math.hypot(X[0] - goal[0], X[1] - goal[1])
    else:
        return [math.hypot(pos[0] - goal[0], pos[1] - goal[1]) for pos in X]


def grid_af_prey(n, pos, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter):
    """
    觅食行为：在邻域内寻找使食物浓度降低的方向
    """
    present_H = lastH[ii]
    rightInf = step_by_iter(cur_iter, MAXGEN, math.sqrt(2))
    allow_area = [p for p in allowed if 0 < distance(pos, p) <= rightInf]
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


def variable_visual_by_iter(cur_iter, max_iter, vis_min=math.sqrt(2), vis_max=4):
    """
    根据当前迭代次数计算可变视野：
      cur_iter：当前迭代次数（从0开始）
      max_iter：最大迭代次数
      vis_min：最小视野（√2）
      vis_max：最大视野（3√2）

    根据公式：
      alpha = exp(-20 * (cur_iter / max_iter) ** 5)
      Vis = alpha * vis_max + (1 - alpha) * vis_min
    """
    ratio = cur_iter / max_iter
    alpha = math.exp(-20 * (ratio ** 5))
    return alpha * vis_max + (1 - alpha) * vis_min


def crowding_factor_by_distance(pos, goal, delta_old, n_f):
    """
    分段拥挤度计算函数（基于鱼离终点的距离）：

    当鱼的位置 pos 与终点 goal 的欧氏距离 d 小于等于 2√2 时，
    采用新公式计算拥挤度：
        δ' = tanh(1/d) / n_f
    （若 d 为 0 则直接返回 1/n_f 以避免除零）

    当 d > 2√2 时，则直接返回预设拥挤度 delta_old。

    参数：
      pos       : 当前鱼的坐标 (例如 (i, j))
      goal      : 终点坐标
      delta_old : 原始或默认的拥挤度
      n_f       : 参数 n_f，用于调节新计算的拥挤度（例如可以设置为鱼的数量或其他常数）

    返回：
      新的拥挤度 δ'
    """
    d = distance(pos, goal)
    threshold = 2 * math.sqrt(2)
    if d <= threshold:
        if d == 0:
            return 1.0 / n_f  # 或其他定义的极限值
        return math.tanh(1.0 / d) / n_f
    else:
        return delta_old


def step_by_iter(cur_iter, max_iter, step):
    """
    根据当前迭代次数计算可变视野：
      cur_iter：当前迭代次数（从0开始）
      max_iter：最大迭代次数
      vis_min：最小视野（√2）
      vis_max：最大视野（3√2）

    根据公式：
      alpha = exp(-20 * (cur_iter / max_iter) ** 5)
      Vis = alpha * vis_max + (1 - alpha) * vis_min
    """
    ratio = cur_iter / max_iter
    alpha = math.exp(-20 * (ratio ** 5))
    return alpha * step + 0.3


def each_af_dist(pos, positions):
    """
    计算 pos 与其他所有鱼之间的距离
    """
    return [distance(pos, other) for other in positions]


def grid_af_follow(n, positions, ii, try_number, lastH, allowed, goal, MAXGEN, delta, cur_iter):
    """
    跟随行为：判断是否向邻近鱼群中局部最优靠拢，否则执行觅食行为
    """
    Xi = positions[ii]
    D = each_af_dist(Xi, positions)
    visual = variable_visual_by_iter(cur_iter, MAXGEN)
    indices = [i for i, d in enumerate(D) if d > 0 and d < visual]
    Nf = len(indices)
    allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
    if Nf > 0:
        Xvisual = [positions[i] for i in indices]
        Hvisual = [lastH[i] for i in indices]
        Xmin = Xvisual[Hvisual.index(min(Hvisual))]
        Hi = lastH[ii]
        delta = crowding_factor_by_distance(Xi, goal, delta, Nf)
        if (min(Hvisual) / Nf) <= (Hi * (1 - delta)):
            for _ in range(try_number):
                if not allow_area:
                    break
                candidate = random.choice(allow_area)
                if distance(candidate, Xmin) < distance(Xi, Xmin):
                    return candidate, grideAF_foodconsistence(candidate, goal)
            return Xi, grideAF_foodconsistence(Xi, goal)
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter)


def grid_af_swarm(n, positions, ii, try_number, lastH, allowed, goal, MAXGEN, delta, cur_iter):
    """
    群聚行为：计算邻近鱼群的质心，判断向质心靠拢是否有利，否则执行觅食行为
    """
    Xi = positions[ii]
    D = each_af_dist(Xi, positions)
    visual = variable_visual_by_iter(cur_iter, MAXGEN)
    indices = [i for i, d in enumerate(D) if d > 0 and d < visual]
    Nf = len(indices)
    allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
    if Nf > 0:
        avg_r = math.ceil(sum(positions[i][0] for i in indices) / Nf)
        avg_c = math.ceil(sum(positions[i][1] for i in indices) / Nf)
        Xc = (avg_r, avg_c)
        Hc = grideAF_foodconsistence(Xc, goal)
        Hi = lastH[ii]
        delta = crowding_factor_by_distance(Xi, goal, delta, Nf)
        if (Hc / Nf) <= (Hi * (1 - delta)):
            for _ in range(try_number):
                if not allow_area:
                    break
                candidate = random.choice(allow_area)
                if distance(candidate, Xc) < distance(Xi, Xc):
                    return candidate, grideAF_foodconsistence(candidate, goal)
            return Xi, grideAF_foodconsistence(Xi, goal)
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, allowed, goal, MAXGEN, cur_iter)


def algorithm_AFSA(env, population_size=20, iterations=100, try_number=6, delta=0.8,
                   random_move_prob=1):
    """
    新版AFSA算法实现：
      1. 每条鱼先尝试跟随和群聚行为，选择其中表现更优者；
      2. 若两者均未能改善当前位置，则采用觅食行为；
      3. 若觅食行为仍未改善（位置未变），则以一定概率随机移动；
      4. 仅在鱼到达目标时更新全局最佳路径 (global_best_cost)；

    参数：
      env: 包含 free_cells、start、goal、rows 等属性的环境对象
      population_size: 鱼群数量
      iterations: 最大迭代次数
      try_number: 每条鱼尝试寻找更优位置的次数
      delta: 拥挤因子
      random_move_prob: 当觅食未改善位置时随机移动概率

    返回：
      当有鱼到达目标时，返回其路径及路径代价；否则返回 (None, inf)
    """
    allowed = sorted(env.free_cells)
    N = population_size
    MAXGEN = iterations

    # 初始化：所有鱼都从起点出发，同时记录每条鱼的轨迹及“食物浓度”
    positions = [env.start for _ in range(N)]
    trajectories = [[env.start] for _ in range(N)]
    H = [grideAF_foodconsistence(env.start, env.goal) for _ in range(N)]

    # 公告牌：全局最佳完整路径（到达目标）的代价记录
    global_best_cost = float('inf')
    global_best_path = None

    # 迭代更新
    for j in range(MAXGEN):
        new_positions = []
        for i in range(N):
            Xi = positions[i]
            current_cost = grideAF_foodconsistence(Xi, env.goal)

            # 1. 尝试跟随与群聚行为
            pos_swarm, cost_swarm = grid_af_swarm(env.rows, positions, i, try_number, H, allowed, env.goal, MAXGEN,
                                                  delta, j)
            pos_follow, cost_follow = grid_af_follow(env.rows, positions, i, try_number, H, allowed, env.goal, MAXGEN,
                                                     delta, j)

            if cost_swarm < current_cost or cost_follow < current_cost:
                # 若两者中有改善，则选择更优的候选位置
                if cost_follow < cost_swarm:
                    candidate, candidate_cost = pos_follow, cost_follow
                else:
                    candidate, candidate_cost = pos_swarm, cost_swarm
            else:
                # 2. 执行觅食行为
                candidate, candidate_cost = grid_af_prey(env.rows, Xi, i, try_number, H, allowed, env.goal, MAXGEN, j)
                # 3. 如果觅食未产生改变，则依据概率随机移动
                if candidate == Xi:
                    if random.random() < random_move_prob:
                        allow_area = [p for p in allowed if 0 < distance(Xi, p) <= math.sqrt(2)]
                        if allow_area:
                            candidate = random.choice(allow_area)
                            candidate_cost = grideAF_foodconsistence(candidate, env.goal)

            new_positions.append(candidate)
            H[i] = candidate_cost
            trajectories[i].append(candidate)

        positions = new_positions.copy()

        # 检查哪些鱼已经到达目标，仅针对它们更新全局最佳路径
        reached = [i for i, pos in enumerate(positions) if pos == env.goal]
        if reached:
            # 遍历到达目标的鱼，比较完整路径代价
            for idx in reached:
                fish_path_cost = path_cost(trajectories[idx])
                if fish_path_cost < global_best_cost:
                    global_best_cost = fish_path_cost
                    global_best_path = trajectories[idx]
            # 既然有鱼到达目标，可以提前结束
            break

    if global_best_path is None:
        return None, float('inf')
    else:
        return global_best_path, global_best_cost


# （2.5）新算法：利用 AFSA 生成初始种群，再由 GA 搜索
def algorithm_GFSA(env, population_size=100, generations=100,
                   crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    """
    新算法：使用 AFSA 生成初始种群，再利用 GA 进行优化
    直接调用 algorithm_GA，并将 population_generator 参数设为 generate_initial_population_afsa
    """
    print("运行基于 AFSA 初始化的 GA 算法...")
    path, cost = algorithm_GA(env, population_size, generations,
                              crossover_rate, mutation_rate, elite_rate,
                              population_generator=generate_initial_population_afsa)
    return path, cost


# -------------------------------
# 3. 绘图辅助函数
# -------------------------------
def plot_paths(env, results, title="路径规划算法对比"):
    """
    绘制环境及各算法路径对比
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='Greys', origin='upper')
    plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(0.5, env.cols, 1), [])
    plt.yticks(np.arange(0.5, env.rows, 1), [])
    plt.title(title)
    plt.plot(env.start[1], env.start[0], 'go', markersize=10, label="起点")
    plt.plot(env.goal[1], env.goal[0], 'mo', markersize=10, label="终点")

    colors = ['r', 'b', 'y', 'c', 'm', 'k', 'orange']
    for idx, (name, path) in enumerate(results.items()):
        if path is not None and len(path) > 0:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            plt.plot(xs, ys, color=colors[idx % len(colors)], marker='o', label=name)
    plt.legend(loc="upper right")
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
    }
    results = {}
    for name, algo in algorithms.items():
        print(f"\n运行算法：{name}")
        path, cost = algo(env)
        print(f"{name} 返回：代价 = {cost}, 路径 = {path}")
        results[name] = path

    plot_paths(env, results, title="统一代价函数与移动方式下的路径规划对比")


if __name__ == "__main__":
    main()
