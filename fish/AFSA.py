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
        (0, 7), (1, 7), (1, 13), (2, 9), (2, 10), (2, 13), (3, 3), (3, 4),
        (3, 7), (3, 8), (3, 13), (4, 8), (5, 1), (5, 7), (5, 10), (5, 12),
        (6, 3), (7, 3), (7, 4), (7, 6), (7, 7), (7, 12), (7, 14), (8, 8),
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
    环境类，封装地图、起点和终点
    """

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape


# 定义允许的移动方向（八个方向）
MOVE_DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]


def get_neighbors(env, cell):
    """
    返回 cell 的所有合法邻居（依据统一移动规则），
    检查边界及障碍；
    对于对角移动，需保证水平和竖直方向邻居均可通行（防止拐角穿越）
    """
    i, j = cell
    neighbors = []
    for d in MOVE_DIRECTIONS:
        ni, nj = i + d[0], j + d[1]
        # 边界检查
        if not (0 <= ni < env.rows and 0 <= nj < env.cols):
            continue
        # 障碍检查
        if env.grid[ni, nj] == 1:
            continue
        # 对角移动时防止拐角穿越
        if d[0] != 0 and d[1] != 0:
            n1 = (i + d[0], j)
            n2 = (i, j + d[1])
            if not (0 <= n1[0] < env.rows and 0 <= n1[1] < env.cols and env.grid[n1] == 0):
                continue
            if not (0 <= n2[0] < env.rows and 0 <= n2[1] < env.cols and env.grid[n2] == 0):
                continue
        neighbors.append((ni, nj))
    return neighbors


def generate_random_path(env, max_steps=150):
    """
    随机生成一条从 env.start 到 env.goal 的路径
    """
    start = env.start
    goal = env.goal
    path = [start]
    visited = set([start])
    current = start
    while current != goal and len(path) < max_steps:
        nbrs = get_neighbors(env, current)
        # 排除已访问节点，尽量避免走回头路
        nbrs = [n for n in nbrs if n not in visited]
        if not nbrs:
            return None
        next_node = random.choice(nbrs)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path if current == goal else None


def generate_random_path_from(env, start, max_steps=150):
    """
    随机生成一条从给定起点到 env.goal 的路径
    """
    goal = env.goal
    path = [start]
    visited = set([start])
    current = start
    while current != goal and len(path) < max_steps:
        nbrs = get_neighbors(env, current)
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
    统一的移动代价函数，根据移动方向 d 返回代价，
    对角移动代价为 sqrt(2)，水平和竖直移动代价为 1
    """
    return np.hypot(d[0], d[1])


def path_cost(path):
    """
    统一的路径代价函数：计算路径中连续点之间的欧氏距离之和
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
    使用八向距离启发函数（兼容统一移动规则）
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)


def astar_planner(env):
    """
    A* 算法接口，输入统一的环境对象，
    返回：路径（列表）和代价
    """
    grid = env.grid
    start, goal = env.start, env.goal
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    # 统一使用 MOVE_DIRECTIONS
    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            # 反向回溯恢复路径
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return path, path_cost(path)

        for d in MOVE_DIRECTIONS:
            neighbor = (current[0] + d[0], current[1] + d[1])
            # 边界与障碍检查
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            if grid[neighbor] == 1:
                continue
            # 对角移动时检查拐角穿越
            if d[0] != 0 and d[1] != 0:
                n1 = (current[0] + d[0], current[1])
                n2 = (current[0], current[1] + d[1])
                if not (0 <= n1[0] < grid.shape[0] and 0 <= n1[1] < grid.shape[1] and grid[n1] == 0):
                    continue
                if not (0 <= n2[0] < grid.shape[0] and 0 <= n2[1] < grid.shape[1] and grid[n2] == 0):
                    continue

            sc = move_cost(d)
            tentative_g = g_score[current] + sc
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return None, float('inf')


def algorithm_AStar(env):
    """
    统一接口：调用 A* 算法模块，返回路径和代价
    """
    print("运行 A* 算法...")
    path, cost = astar_planner(env)
    print("A* 路径代价:", cost)
    return path, cost


# （2.2）RRT 算法模块
# -------------------------------
# 公共函数：计算两点间欧氏距离
# -------------------------------
def distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def get_neighbors_rrt(node, target, L=1.4):
    """
    根据当前节点 node 和采样目标 target 生成新节点，新节点离 node 的距离约为步长 L。
    如果生成的新节点与 node 相同，则在目标方向上至少移动 1 个单位。
    """
    dx = target[0] - node[0]
    dy = target[1] - node[1]
    dist = np.hypot(dx, dy)
    if dist == 0:
        return node
    new_x = int(round(node[0] + (dx / dist) * L))
    new_y = int(round(node[1] + (dy / dist) * L))
    new_node = (new_x, new_y)
    if new_node == node:
        new_node = (node[0] + int(np.sign(dx)), node[1] + int(np.sign(dy)))
    return new_node


# -------------------------------
# RRT 算法模块：单次 RRT 规划
# -------------------------------
def rrt_planner(env, max_iter=5000, max_sample_attempts=100):
    """
    RRT 算法接口：在给定环境 env 内从起点生成一棵搜索树，
    当有树节点距离目标足够近时，回溯生成路径。

    参数：
      env: GridEnvironment 对象
      max_iter: 最大迭代次数
      max_sample_attempts: 每次采样的最大尝试次数

    返回：
      若成功，返回从起点到终点的路径（列表）；否则返回 None。
    """
    grid = env.grid
    start = env.start
    goal = env.goal
    tree = {start: None}
    nodes = [start]

    for _ in range(max_iter):
        # 多次尝试采样一个非障碍点
        for _ in range(max_sample_attempts):
            rand = (random.randint(0, env.rows - 1),
                    random.randint(0, env.cols - 1))
            if grid[rand] == 0:
                break
        else:
            continue  # 连续多次采样失败则跳过当前迭代

        # 从已有节点中选取离采样点最近的节点
        nearest = min(nodes, key=lambda n: distance(n, rand))
        new_node = get_neighbors_rrt(nearest, rand)

        # 检查新节点是否在合法区域内
        if not (0 <= new_node[0] < env.rows and 0 <= new_node[1] < env.cols):
            continue
        if grid[new_node] == 1 or new_node in tree:
            continue

        # 对角移动时检查拐角穿越：检测两个邻接节点是否均无障碍
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

        # 判断是否达到目标附近，如果新节点足够接近目标，则回溯生成路径
        if distance(new_node, goal) <= 1:
            if new_node == goal:
                tree[goal] = nearest
            else:
                tree[goal] = new_node
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = tree[node]
            return path[::-1]
    return None

# -------------------------------
# 统一接口：多次运行 RRT 选择最优路径
# -------------------------------
def algorithm_RRT(env, n_runs=100):
    """
    统一接口：多次执行 rrt_planner，选出总代价最低的路径作为最终解。

    参数：
      env: GridEnvironment 对象
      n_runs: RRT 规划的运行次数（默认 100 次）

    返回：
      (最佳路径, 最佳路径代价)
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


# -------------------------------
# 3. 遗传算法模块（GA）
# -------------------------------
def algorithm_GA(env, population_size=100, generations=100,
                 crossover_rate=0.8, mutation_rate=0.05, elite_rate=0.05):
    """
    遗传算法模块，用于在给定环境 env 上搜索路径。

    参数：
      env: 环境对象，包含 grid, start, goal, rows, cols
      population_size: 种群大小（默认 100）
      generations: 进化代数（默认 100）
      crossover_rate: 交叉概率（默认 0.8）
      mutation_rate: 变异概率（默认 0.05）
      elite_rate: 精英保留比例（默认 0.05）

    返回：
      最优路径（列表）和对应的代价（路径总代价）
    """
    # 初始化种群：调用 generate_random_path(env)
    population = []
    while len(population) < population_size:
        p = generate_random_path(env)
        if p is not None:
            population.append(p)

    best_solution = None
    best_cost = float('inf')

    def tournament_selection(pop, fitnesses, tournament_size=2):
        """
        锦标赛选择：从种群中随机选择若干个体，取代价最低的个体。
        """
        selected = []
        for _ in range(len(pop)):
            contenders = random.sample(list(zip(pop, fitnesses)), tournament_size)
            winner = min(contenders, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(parent1, parent2):
        """
        交叉操作：寻找两个父代中除起点与终点外的公共节点，
        以该公共节点为交叉点交换后段信息，生成两个子代。
        如果没有公共节点，则直接返回父代。
        """
        common_nodes = set(parent1[1:-1]) & set(parent2[1:-1])
        if common_nodes:
            common_node = random.choice(list(common_nodes))
            idx1 = parent1.index(common_node)
            idx2 = parent2.index(common_node)
            child1 = parent1[:idx1] + parent2[idx2:]
            child2 = parent2[:idx2] + parent1[idx1:]
            if child1[0] != env.start or child1[-1] != env.goal:
                child1 = parent1
            if child2[0] != env.start or child2[-1] != env.goal:
                child2 = parent2
            return child1, child2
        else:
            return parent1, parent2

    def mutate(individual):
        """
        变异操作：随机选择个体中非起点和终点位置作为变异点，
        利用 generate_random_path_from(env, node) 重生成从该点到终点的子路径，
        并拼接成新的个体。
        """
        if len(individual) <= 2:
            return individual
        mut_idx = random.randint(1, len(individual) - 2)
        new_subpath = generate_random_path_from(env, individual[mut_idx])
        if new_subpath is not None:
            mutated = individual[:mut_idx] + new_subpath[1:]
            return mutated
        return individual

    # 主进化循环
    for gen in range(generations):
        fitnesses = [path_cost(ind) for ind in population]
        # 更新全局最优解
        for ind, cost in zip(population, fitnesses):
            if cost < best_cost:
                best_cost = cost
                best_solution = ind

        selected = tournament_selection(population, fitnesses, tournament_size=2)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            if random.random() < crossover_rate:
                child1, child2 = crossover(p1, p2)
            else:
                child1, child2 = p1, p2
            offspring.extend([child1, child2])
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i])
        elite_count = max(1, int(elite_rate * population_size))
        sorted_population = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0])]
        new_population = sorted_population[:elite_count] + offspring[:population_size - elite_count]
        population = new_population

        print(f"GA Generation {gen + 1}: Best cost = {best_cost}")

    return best_solution, best_cost


def algorithm_AFSA(env, population_size=100, iterations=100, try_number=10, random_move_prob=0.3):
    """
    人工鱼群算法（AFSA）模块，用于在给定环境 env 上搜索路径。

    参数：
      env: GridEnvironment 对象，包含 grid, start, goal, rows, cols
      population_size: 鱼群中鱼的数量（初始种群大小），默认 30
      iterations: 迭代次数，默认 100
      try_number: 每条鱼在视觉范围内尝试查找更优解的次数，默认 5 次
      random_move_prob: 当鱼未能在视觉范围内找到更优解时，进行随机移动的概率，默认 0.3

    返回：
      (最佳路径, 最佳路径代价)

    依赖：
      - generate_random_path(env): 随机生成一条从起点到终点的路径
      - generate_random_path_from(env, start): 从给定起点随机生成一条路径到终点
      - path_cost(path): 计算路径的总代价（连续点间欧氏距离之和）
    """
    # 初始化鱼群，每条鱼是一条候选路径（利用随机路径生成函数）
    population = []
    while len(population) < population_size:
        candidate = generate_random_path(env)
        if candidate is not None:
            population.append(candidate)

    best_solution = None
    best_cost = float('inf')

    # 主迭代过程
    for it in range(iterations):
        # 对鱼群中每条鱼更新（局部搜索或随机移动）
        for i in range(population_size):
            current_path = population[i]
            current_cost = path_cost(current_path)
            improved = False

            # 在规定的尝试次数内搜索视觉邻域
            for _ in range(try_number):
                # 扰动策略：随机选择当前路径中一个非起点与非终点节点作为扰动点，
                # 然后利用 generate_random_path_from 从该点重新生成一段路径
                if len(current_path) <= 2:
                    candidate_path = current_path
                else:
                    idx = random.randint(1, len(current_path) - 2)
                    new_segment = generate_random_path_from(env, current_path[idx])
                    if new_segment is None:
                        candidate_path = current_path
                    else:
                        candidate_path = current_path[:idx] + new_segment[1:]
                candidate_cost = path_cost(candidate_path)

                # 如果发现候选解更优，则鱼前进到此解并退出当前尝试
                if candidate_cost < current_cost:
                    population[i] = candidate_path
                    current_path = candidate_path
                    current_cost = candidate_cost
                    improved = True
                    break  # 找到一个更优解后停止在视觉领域继续搜索

            # 若经过视觉搜索后未找到更优解，则按一定概率进行随机移动（跳出局部最优）
            if not improved and random.random() < random_move_prob:
                candidate = generate_random_path(env)
                if candidate is not None:
                    population[i] = candidate
                    current_path = candidate
                    current_cost = path_cost(candidate)

        # 每轮迭代后更新全局最优解
        for candidate in population:
            c = path_cost(candidate)
            if c < best_cost:
                best_cost = c
                best_solution = candidate

        print(f"AFSA 迭代 {it + 1}: 最佳路径代价 = {best_cost}")

    return best_solution, best_cost


# -------------------------------
# 4. 绘图辅助函数（对比展示）
# -------------------------------
def plot_paths(env, results, title="路径规划算法对比"):
    """
    绘制环境及各算法路径对比；
    :param env: GridEnvironment 对象
    :param results: 字典 {算法名称: 路径列表}
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='Greys', origin='upper')
    plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(0.5, env.cols, 1), [])
    plt.yticks(np.arange(0.5, env.rows, 1), [])
    plt.title(title)

    # 绘制起点和终点
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
# 5. 主函数
# -------------------------------
def main():
    # 生成环境
    grid, start, goal = generate_fixed_grid()
    env = GridEnvironment(grid, start, goal)

    # 定义各算法接口，此处保证 GA 被正确调用
    algorithms = {
        "AStar": algorithm_AStar,
        "RRT": algorithm_RRT,
        "GA": algorithm_GA,
        "AFSA": algorithm_AFSA
    }

    results = {}
    for name, algo_func in algorithms.items():
        print(f"\n运行算法：{name}")
        path, cost = algo_func(env)
        print(f"{name} 返回：代价 = {cost}, 路径 = {path}")
        results[name] = path

    plot_paths(env, results, title="统一代价函数与移动方式下的路径规划对比")


if __name__ == "__main__":
    main()
