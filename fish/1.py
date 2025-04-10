import numpy as np
import matplotlib.pyplot as plt
import math, random, time


# ---------------- 环境类及移动方向 ----------------
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


# ---------------- 固定地图的生成 ----------------
def generate_fixed_grid():
    """
    生成固定地图（15×15）和障碍布局：
      - grid：15×15 的 0-1 矩阵，0 表示通行区域，1 表示障碍
      - start：起点 (0, 0)
      - goal：终点 (14, 14)
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


# ---------------- 工具函数 ----------------
def distance(p1, p2):
    """
    计算两个二维坐标 p1、p2 之间的欧氏距离
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def grideAF_foodconsistence(X, goal):
    """
    计算“食物浓度”，用当前位置到目标位置的欧氏距离表示。
    参数 X 可为单个位置（元组）或多个位置（列表）
    """
    if isinstance(X, tuple):
        return distance(X, goal)
    else:
        return [distance(pos, goal) for pos in X]


def allowed_positions(n, Barrier):
    """
    返回在 0~n-1 范围内，除去障碍物位置（Barrier）以外的所有位置，位置以 (row, col) 表示
    """
    all_positions = [(i, j) for i in range(n) for j in range(n)]
    return sorted([pos for pos in all_positions if pos not in Barrier])


# ---------------- 鱼群算法中的行为函数 ----------------
def grid_af_prey(n, pos, ii, try_number, lastH, Barrier, goal, MAXGEN):
    """
    觅食行为：当前鱼尝试在邻域内寻找能使食物浓度降低的移动方向，
    返回 (nextPosition, nextPositionH)
    """
    present_H = lastH[ii]
    rightInf = math.sqrt(2)
    rightInf = math.ceil(rightInf * (1 - 1 / MAXGEN))
    allow = allowed_positions(n, Barrier)
    allow_area = []
    for p in allow:
        d = distance(pos, p)
        if d > 0 and d <= rightInf:
            allow_area.append(p)
    m = random.choices([0, 1], weights=[0.2, 0.8])[0]
    nextPosition = None
    if m == 0:
        for _ in range(try_number):
            if not allow_area:
                break
            Xj = random.choice(allow_area)
            Hj = grideAF_foodconsistence(Xj, goal)
            if present_H > Hj:
                nextPosition = Xj
                break
    else:
        H_min = present_H
        for p in allow_area:
            Hi = grideAF_foodconsistence(p, goal)
            if Hi < H_min:
                H_min = Hi
                nextPosition = p
    if nextPosition is None:
        if allow_area:
            nextPosition = random.choice(allow_area)
        else:
            nextPosition = pos
    nextPositionH = grideAF_foodconsistence(nextPosition, goal)
    return nextPosition, nextPositionH


def each_af_dist(pos, positions):
    """
    计算当前鱼（pos）与所有鱼（positions）之间的距离列表
    """
    return [distance(pos, other) for other in positions]


def grid_af_follow(n, positions, ii, try_number, lastH, Barrier, goal, MAXGEN, delta):
    """
    跟随行为：根据邻近鱼的状态判断是否向局部最优位置靠拢，否则执行觅食行为
    """
    Xi = positions[ii]
    D = each_af_dist(Xi, positions)
    visual = np.mean(D)
    indices = [i for i, d in enumerate(D) if d > 0 and d < visual]
    Nf = len(indices)
    allow = allowed_positions(n, Barrier)
    allow_area = [p for p in allow if distance(Xi, p) > 0 and distance(Xi, p) <= math.sqrt(2)]
    if Nf > 0:
        Xvisual = [positions[i] for i in indices]
        Hvisual = [lastH[i] for i in indices]
        min_index = Hvisual.index(min(Hvisual))
        Xmin = Xvisual[min_index]
        Hi = lastH[ii]
        if (min(Hvisual) / Nf) <= (Hi * delta):
            for _ in range(try_number):
                if not allow_area:
                    break
                Xnext = random.choice(allow_area)
                if distance(Xnext, Xmin) < distance(Xi, Xmin):
                    nextPosition = Xnext
                    nextPositionH = grideAF_foodconsistence(nextPosition, goal)
                    return nextPosition, nextPositionH
            nextPosition = Xi
            nextPositionH = grideAF_foodconsistence(nextPosition, goal)
            return nextPosition, nextPositionH
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, Barrier, goal, MAXGEN)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, Barrier, goal, MAXGEN)


def grid_af_swarm(n, positions, ii, try_number, lastH, Barrier, goal, MAXGEN, delta):
    """
    群聚行为：计算邻近鱼的位置质心，并判断移动是否能改善食物浓度，
    否则执行觅食行为
    """
    Xi = positions[ii]
    D = each_af_dist(Xi, positions)
    visual = np.mean(D)
    indices = [i for i, d in enumerate(D) if d > 0 and d < visual]
    Nf = len(indices)
    allow = allowed_positions(n, Barrier)
    allow_area = [p for p in allow if distance(Xi, p) > 0 and distance(Xi, p) <= math.sqrt(2)]
    if Nf > 0:
        sum_r = sum(positions[i][0] for i in indices)
        sum_c = sum(positions[i][1] for i in indices)
        avg_r = math.ceil(sum_r / Nf)
        avg_c = math.ceil(sum_c / Nf)
        Xc = (avg_r, avg_c)
        Hc = grideAF_foodconsistence(Xc, goal)
        Hi = lastH[ii]
        if (Hc / Nf) <= (Hi * delta):
            for _ in range(try_number):
                if not allow_area:
                    break
                Xnext = random.choice(allow_area)
                if distance(Xnext, Xc) < distance(Xi, Xc):
                    nextPosition = Xnext
                    nextPositionH = grideAF_foodconsistence(nextPosition, goal)
                    return nextPosition, nextPositionH
            nextPosition = Xi
            nextPositionH = grideAF_foodconsistence(nextPosition, goal)
            return nextPosition, nextPositionH
        else:
            return grid_af_prey(n, Xi, ii, try_number, lastH, Barrier, goal, MAXGEN)
    else:
        return grid_af_prey(n, Xi, ii, try_number, lastH, Barrier, goal, MAXGEN)


def draw_path(path):
    """
    根据给定的二维坐标路径绘图
    """
    xs = [pos[1] + 0.5 for pos in path]
    ys = [pos[0] + 0.5 for pos in path]
    plt.figure()
    plt.plot(xs, ys, 'r-', linewidth=2)
    plt.title("Path found by AFSA")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# ---------------- 主函数 ----------------
def main():
    # 生成固定地图，并由 GridEnvironment 封装
    grid, start, goal = generate_fixed_grid()
    env = GridEnvironment(grid, start, goal)

    # 生成障碍集合 Barrier（障碍以 (row, col) 表示）
    Barrier = [(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r, c] == 1]

    # 显示地图（使用 np.pad 对地图边界进行补白）
    padded_grid = np.pad(env.grid, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    plt.figure()
    plt.pcolor(padded_grid, cmap='gray', edgecolors='k', linewidths=0.5)
    plt.title("Fixed 15×15 Grid Map")
    plt.xticks(np.arange(0.5, env.rows + 0.5, 10))
    plt.yticks(np.arange(0.5, env.rows + 0.5, 10))
    plt.gca().invert_yaxis()
    plt.show()

    # 参数设置
    N = 50  # 人工鱼数量
    try_number = 8  # 尝试步数
    MAXGEN = 50  # 最大迭代次数
    delta = 0.618  # 拥挤因子
    shiftFreq = 4  # 行为切换频率

    # 初始所有鱼的位置均为起点（使用二维坐标表示）
    ppValue = [env.start for _ in range(N)]
    trajectories = [[env.start] for _ in range(N)]
    H = grideAF_foodconsistence(ppValue, env.goal)  # 列表形式的食物浓度

    count = 1
    positions_history = []
    positions_history.append(ppValue.copy())

    start_time = time.time()

    # 主迭代循环
    for j in range(MAXGEN):
        new_positions = []
        if count % shiftFreq == 0:
            shift = 2
        else:
            shift = 1

        if shift == 1:
            # 触发觅食行为
            for i in range(N):
                pos, posH = grid_af_prey(env.rows, ppValue[i], i, try_number, H, Barrier, env.goal, MAXGEN)
                new_positions.append(pos)
                H[i] = posH
                trajectories[i].append(pos)
        else:
            # 同时执行群聚与跟随行为，选择较优的那个
            for i in range(N):
                pos_swarm, posH_swarm = grid_af_swarm(env.rows, ppValue, i, try_number, H, Barrier, env.goal, MAXGEN,
                                                      delta)
                pos_follow, posH_follow = grid_af_follow(env.rows, ppValue, i, try_number, H, Barrier, env.goal, MAXGEN,
                                                         delta)
                if posH_follow < posH_swarm:
                    pos, posH = pos_follow, posH_follow
                else:
                    pos, posH = pos_swarm, posH_swarm
                new_positions.append(pos)
                H[i] = posH
                trajectories[i].append(pos)
        count += 1
        ppValue = new_positions.copy()
        positions_history.append(ppValue.copy())
        reached = [i for i, pos in enumerate(ppValue) if pos == env.goal]
        if reached:
            print("有鱼在第", j + 1, "次迭代中到达终点。")
            break
    else:
        print("没有鱼达到目标点！")
        return

    # 在到达终点的鱼中，选择路径最短的鱼
    path_lengths = []
    for i in reached:
        traj = trajectories[i]
        length_sum = 0
        for k in range(len(traj) - 1):
            length_sum += distance(traj[k], traj[k + 1])
        path_lengths.append(length_sum)
    best_index = reached[path_lengths.index(min(path_lengths))]
    best_path = trajectories[best_index]
    print("最短路径长度为: {:.2f}".format(min(path_lengths)))
    draw_path(best_path)

    elapsed = time.time() - start_time
    print("算法运行时间: {:.2f} 秒".format(elapsed))


if __name__ == '__main__':
    main()
