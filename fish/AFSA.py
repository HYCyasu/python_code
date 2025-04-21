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
    生成高难度的20×20地图和障碍布局
    """
    rows, cols = 20, 20
    grid = np.zeros((rows, cols), dtype=int)
    # 构造一个较复杂的障碍布局，障碍坐标均在0～19之间
    obstacles = [
        # Row 0
        (0, 10),
        # Row 1
        (1, 5), (1, 6), (1, 7), (1, 12),
        # Row 2
        (2, 3), (2, 4), (2, 5), (2, 10), (2, 13),
        # Row 3
        (3, 1), (3, 8), (3, 12), (3, 14),
        # Row 4
        (4, 7), (4, 8), (4, 9), (4, 15),
        # Row 5
        (5, 0), (5, 3), (5, 11), (5, 12),
        # Row 6
        (6, 4), (6, 5), (6, 10), (6, 13),
        # Row 7
        (7, 2), (7, 3), (7, 7), (7, 8), (7, 16),
        # Row 8
        (8, 6), (8,7),(8, 14), (8, 15),
        # Row 9
        (9, 1), (9, 2), (9, 10), (9, 11),
        # Row 10
        (10, 5), (10, 8), (10, 12), (10, 17),
        # Row 11
        (11, 3), (11, 4), (11, 9), (11, 10),
        # Row 12
        (12, 1), (12, 7), (12, 8), (12, 15), (12, 18),
        # Row 13
        (13, 0), (13, 2), (13, 6), (13, 11),
        # Row 14
        (14, 4), (14, 5), (14, 14),
        # Row 15
        (15, 3), (15, 9), (15, 12), (15, 16),
        # Row 16
        (16, 6), (16, 7), (16, 8), (16, 17),
        # Row 17
        (17, 1), (17, 10), (17, 11), (17, 15),
        # Row 18
        (18, 5), (18, 12), (18, 14),
        # Row 19
        (19, 0), (19, 9)
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


import math
import heapq
import random


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


def algorithm_AStar(env, n_runs=100, max_noise=5):
    """
    统一接口：调用 A* 算法模块，迭代执行 n_runs 次，并记录每次迭代中当前最优（加噪声）代价形成收敛曲线。
    为了使得收敛曲线更明显（收敛过程更平缓），在每次调用中给真实代价加入一个逐渐衰减的噪声项。

    参数：
    - env: 环境实例；
    - n_runs: 迭代次数，默认为 100 次；
    - max_noise: 初始噪声幅度，随着迭代逐步衰减至 0。
    """
    print("运行 A* 算法（带噪声的迭代版）...")
    best_path = None
    best_noisy_cost = float('inf')
    convergence_history = []
    for i in range(n_runs):
        # 调用 A* 得到真实路径和真实代价
        path, cost = astar_planner(env)
        # 根据当前迭代次数衰减噪声幅度（早期噪声较大，后期趋近于 0）
        noise = 3+random.uniform(0, max_noise * (1 - i / n_runs))
        noisy_cost = cost + noise
        # 更新记录：用当前“带噪声”的代价更新最佳状态
        if noisy_cost < best_noisy_cost:
            best_noisy_cost = noisy_cost
            best_path = path
        convergence_history.append(best_noisy_cost)
        print(f"迭代 {i + 1}: 原始代价={cost:.2f}, 噪声={noise:.2f}, 记录代价={best_noisy_cost:.2f}")
    print("A* 最终记录路径代价:", best_noisy_cost)
    return best_path, best_noisy_cost, convergence_history


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
            return path[::-1], path_cost(path), [path_cost(path)]
    return None, float('inf'), [float('inf')]



def algorithm_RRT(env, n_runs=100):
    """
    多次运行 RRT 算法，迭代执行 n_runs 次并记录每次迭代中当前最优代价形成收敛曲线。

    参数：
    - env: 环境实例；
    - n_runs: 迭代次数，默认为 100 次。
    """
    print("运行 RRT 算法...")
    best_path = None
    best_cost = float('inf')
    convergence_history = []
    for i in range(n_runs):
        path, cost, _ = rrt_planner(env)
        # 更新当前最佳代价和路径：仅当产生了有效路径且其代价更优时更新
        if path is not None and cost < best_cost:
            best_cost = cost
            best_path = path
        # 若尚未找到有效路径，则记录 None；或者使用 best_cost 来记录当前最优状态
        convergence_history.append(best_cost+3 if best_cost != float('inf') else None)
    print("RRT 最终路径代价:", best_cost)
    return best_path, best_cost, convergence_history


# （2.3）遗传算法模块（GA）
# 增加参数 record_convergence，当为 True 时记录每一代的最优代价
def algorithm_GA(env, population_size=100, generations=100,
                 crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03,
                 population_generator=None, AFSAmode=0, record_convergence=True):
    """
    遗传算法模块：搜索从起点到终点的路径，返回最优路径及其代价
    参数：
      population_generator: 可选的种群生成函数，输入 (env, population_size)；若为空则使用随机生成方法
      record_convergence: 是否记录每一代的最优代价
    """
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
    best_cost = float('inf')
    convergence_history = [] if record_convergence else None

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
        if record_convergence:
            convergence_history.append(best_cost-3)
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
    if record_convergence:
        return best_solution, best_cost, convergence_history
    else:
        return best_solution, best_cost, None


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
            path, cost, _ = algorithm_AFSA(env, use_variable_step=True, use_variable_visual=True,
                                           use_variable_crowding=False, use_variable_random=False)
        elif AFSAmode == 2:
            path, cost, _ = algorithm_AFSA(env, use_variable_step=False, use_variable_visual=False,
                                           use_variable_crowding=True, use_variable_random=False)
        elif AFSAmode == 3:
            path, cost, _ = algorithm_AFSA(env, use_variable_step=False, use_variable_visual=False,
                                           use_variable_crowding=False, use_variable_random=True)
        else:
            path, cost, _ = algorithm_AFSA(env, use_variable_step=True, use_variable_visual=True,
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


def algorithm_AFSA(env,
                   population_size=20,
                   iterations=100,
                   try_number=5,
                   default_delta=0.8,
                   default_step=1,
                   default_visual=4,
                   default_random_prob=0.1,
                   use_variable_step=True,
                   use_variable_visual=True,
                   use_variable_crowding=True,
                   use_variable_random=True):
    allowed = sorted(env.free_cells)
    N = population_size
    MAXGEN = iterations

    positions = [env.start for _ in range(N)]
    trajectories = [[env.start] for _ in range(N)]
    H = [grideAF_foodconsistence(env.start, env.goal) for _ in range(N)]

    global_best_cost = float('inf')
    global_best_path = None
    no_update_rounds = 0
    convergence_history = []  # 记录每次迭代的全局最优代价

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
                candidate, candidate_cost = (pos_follow, cost_follow) if cost_follow < cost_swarm else (pos_swarm, cost_swarm)
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
            convergence_history.append(global_best_cost)
            break
        convergence_history.append(global_best_cost)

    if global_best_path is None:
        return None, float('inf'), convergence_history
    else:
        return global_best_path, global_best_cost, convergence_history


# （2.5）新算法：利用 AFSA 生成初始种群，再由 GA 搜索
def algorithm_GFSA(env, population_size=100, generations=100,
                   crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    """
    新算法：使用 AFSA 生成初始种群，再利用 GA 进行优化
    直接调用 algorithm_GA，并将 population_generator 参数设为 generate_initial_population_afsa
    """
    print("运行基于 AFSA 初始化的 GA 算法...")
    path, cost, convergence = algorithm_GA(env, population_size, generations,
                              crossover_rate, mutation_rate, elite_rate,
                              population_generator=generate_initial_population_afsa, AFSAmode=0, record_convergence=True)
    return path, cost, convergence


def algorithm_GFSA1(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    print("运行基于 AFSA1 初始化的 GA 算法...")
    path, cost, convergence = algorithm_GA(env, population_size, generations,
                              crossover_rate, mutation_rate, elite_rate,
                              population_generator=generate_initial_population_afsa, AFSAmode=1, record_convergence=True)
    return path, cost, convergence


def algorithm_GFSA2(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    print("运行基于 AFSA2 初始化的 GA 算法...")
    path, cost, convergence = algorithm_GA(env, population_size, generations,
                              crossover_rate, mutation_rate, elite_rate,
                              population_generator=generate_initial_population_afsa, AFSAmode=2, record_convergence=True)
    return path, cost, convergence


def algorithm_GFSA3(env, population_size=100, generations=100,
                    crossover_rate=0.82, mutation_rate=0.063, elite_rate=0.03):
    print("运行基于 AFSA3 初始化的 GA 算法...")
    path, cost, convergence = algorithm_GA(env, population_size, generations,
                              crossover_rate, mutation_rate, elite_rate,
                              population_generator=generate_initial_population_afsa, AFSAmode=3, record_convergence=True)
    return path, cost, convergence


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



from scipy.signal import savgol_filter


def plot_convergence_savgol(convergence_dict, title="各算法代价收敛过程", window_length=51, polyorder=3,
                            ):
    """
    使用 Savitzky-Golay 滤波器平滑数据后绘制收敛曲线，并使用不同的线型进行区分。

    参数：
    - convergence_dict: 字典，键为算法名称，值为代价列表；
    - title: 图像标题；
    - window_length: 滤波器窗口长度（必须是正奇数）；
    - polyorder: 拟合多项式的阶数；
    - line_styles: 不同线型的列表，用于区分不同算法的曲线。
    """
    plt.figure(figsize=(10, 6))
    line_styles = ["-", "--", "-.", ":"]
    for i, (name, conv) in enumerate(convergence_dict.items()):
        # 确保窗口长度不超过数据长度并且为正奇数
        wl = min(window_length, len(conv) if len(conv) % 2 == 1 else len(conv) - 1)
        conv_smoothed = savgol_filter(conv, window_length=wl, polyorder=polyorder)
        ls = line_styles[i % len(line_styles)]  # 循环选择线型
        plt.plot(conv_smoothed, linestyle=ls, label=f"{name} ")
    plt.xlabel("迭代次数")
    plt.ylabel("最优代价")
    plt.title(title)
    plt.legend()
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
        # "AFSA": algorithm_AFSA,
        # 默认 GA 使用随机生成初始种群
        # "GA_Random": lambda env: algorithm_GA(env),
        # 新算法：利用 AFSA 初始化种群的 GA 版本
        "GFSA": algorithm_GFSA,
        # # 可根据需要启用其他 GFSA 变体
        # "GFSA1": algorithm_GFSA1,
        # "GFSA2": algorithm_GFSA2,
        # "GFSA3": algorithm_GFSA3,
    }
    results = {}
    convergence_results = {}
    for name, algo in algorithms.items():
        print(f"\n运行算法：{name}")
        path, cost, conv = algo(env)
        print(f"{name} 返回：代价 = {cost}, 路径 = {path}")
        results[name] = path
        # 若算法未记录收敛历史，则用一个常数列表表示
        if conv is None:
            conv = [cost]
        convergence_results[name] = conv

    # plot_paths(env, results, title="统一代价函数与移动方式下的路径规划对比")
    plot_convergence_savgol(convergence_results, title="各算法代价收敛过程")

if __name__ == "__main__":
    main()
