import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import matplotlib

# 使用支持中文的字体（如果需要中文显示）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 生成地图
# -------------------------------
rows, cols = 15, 15
grid = np.zeros([rows, cols], dtype=int)

obstacles = [
    (0,7), (1,7), (1,13), (2,9), (2,10), (2,13), (3,3), (3,4),
    (3,7), (3,8), (3,13), (4,8), (5,1), (5,7), (5,10), (5,12),
    (6,3), (7,3), (7,4), (7,6), (7,7), (7,12), (7,14), (8,8),
    (8,11), (8,12), (9,2), (10,2), (10,4), (10,13), (11,8),
    (11,9), (11,10), (12,3), (12,4), (12,6), (12,10), (13,0),
    (13,4), (13,6)
]

for r, c in obstacles:
    grid[r, c] = 1

# -------------------------------
# 定义启发函数（八向距离）
# -------------------------------
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

# -------------------------------
# A* 算法（允许八向移动，但禁止拐角穿越）
# -------------------------------
def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            return path[::-1]

        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            # 检查边界
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            # 检查障碍物
            if grid[neighbor] == 1:
                continue
            # 仅在对角移动时（两个分量都不为0）检查拐角穿越
            if d[0] != 0 and d[1] != 0:
                neighbor1 = (current[0] + d[0], current[1])
                neighbor2 = (current[0], current[1] + d[1])
                if not (0 <= neighbor1[0] < grid.shape[0] and 0 <= neighbor1[1] < grid.shape[1] and grid[neighbor1] == 0):
                    continue
                if not (0 <= neighbor2[0] < grid.shape[0] and 0 <= neighbor2[1] < grid.shape[1] and grid[neighbor2] == 0):
                    continue

            move_cost = np.hypot(d[0], d[1])
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return None  # 未找到路径

# -------------------------------
# RRT 算法（允许八向移动，并禁止拐角穿越）
# -------------------------------
def distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def get_neighbors_rrt(node, target, L=1.4):
    dx = target[0] - node[0]
    dy = target[1] - node[1]
    dist = np.hypot(dx, dy)
    if dist == 0:
        return node
    new_x = int(round(node[0] + (dx / dist) * L))
    new_y = int(round(node[1] + (dy / dist) * L))
    new_node = (new_x, new_y)
    # 如果新节点与当前节点相同，则向目标方向至少移动一个单位
    if new_node == node:
        new_node = (node[0] + int(np.sign(dx)), node[1] + int(np.sign(dy)))
    return new_node

def rrt(grid, start, goal, max_iter=5000, max_sample_attempts=100):
    tree = {start: None}
    nodes = [start]

    for _ in range(max_iter):
        for _ in range(max_sample_attempts):
            rand = (random.randint(0, grid.shape[0] - 1),
                    random.randint(0, grid.shape[1] - 1))
            if grid[rand] == 0:
                break
        else:
            continue

        nearest = min(nodes, key=lambda n: distance(n, rand))
        new_node = get_neighbors_rrt(nearest, rand)

        if not (0 <= new_node[0] < grid.shape[0] and 0 <= new_node[1] < grid.shape[1]):
            continue
        if grid[new_node] == 1 or new_node in tree:
            continue

        dx = new_node[0] - nearest[0]
        dy = new_node[1] - nearest[1]
        if dx != 0 and dy != 0:
            neighbor1 = (nearest[0] + dx, nearest[1])
            neighbor2 = (nearest[0], nearest[1] + dy)
            if (0 <= neighbor1[0] < grid.shape[0] and 0 <= neighbor1[1] < grid.shape[1] and grid[neighbor1] == 1) or \
               (0 <= neighbor2[0] < grid.shape[0] and 0 <= neighbor2[1] < grid.shape[1] and grid[neighbor2] == 1):
                continue

        tree[new_node] = nearest
        nodes.append(new_node)

        if distance(new_node, goal) <= 1.0:
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
# 计算路径代价函数（欧氏距离之和）
# -------------------------------
def path_cost(path):
    cost = 0.0
    for i in range(1, len(path)):
        cost += np.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
    return cost

# -------------------------------
# 执行 A* 与 RRT 算法
# -------------------------------
start = (0, 0)
goal = (rows - 1, cols - 1)
path_astar = astar(grid, start, goal)
astar_cost = path_cost(path_astar) if path_astar is not None else float('inf')

n_runs = 100
best_rrt = None
best_rrt_cost = float('inf')
for _ in range(n_runs):
    path = rrt(grid, start, goal)
    if path is not None:
        cost = path_cost(path)
        if cost < best_rrt_cost:
            best_rrt_cost = cost
            best_rrt = path

print("A* cost:", astar_cost)
print("Best RRT cost:", best_rrt_cost)

# -------------------------------
# 绘制结果（使用网格线显示，点处于格子中间）
# -------------------------------
plt.figure(figsize=(15, 15))
# 直接用 imshow 显示 grid（无 extent），用默认坐标范围 [0, cols) 与 [0, rows)
plt.imshow(grid, cmap='Greys', origin='upper')
# 设置网格线在每个单元格边缘处
plt.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1)
plt.xticks(np.arange(0.5, cols, 1), [])
plt.yticks(np.arange(0.5, rows, 1), [])
plt.title("15x15 Grid with Obstacles")

# 绘制 A* 路径（坐标加 0.5，使其位于单元格中心；注意 x 对应列，y 对应行）
if path_astar is not None:
    path_astar_np = np.array(path_astar)
    plt.plot(path_astar_np[:, 1], path_astar_np[:, 0], 'r-', linewidth=2, label="A*")
else:
    print("A* 未找到路径")

# 绘制最佳 RRT 路径
if best_rrt is not None:
    best_rrt_np = np.array(best_rrt)
    plt.plot(best_rrt_np[:, 1], best_rrt_np[:, 0], 'b-', linewidth=2, label="RRT")
else:
    print("RRT 未找到路径")

# 绘制起点和终点
plt.plot(start[1], start[0], 'go', markersize=10, label="起点")
plt.plot(goal[1], goal[0], 'mo', markersize=10, label="终点")
plt.legend(loc="upper right")
plt.show()
