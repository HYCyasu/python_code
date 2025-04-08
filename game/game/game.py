import numpy as np
import cvxpy as cp

# ==== 1. 设定 QP 参数 ====
Q = np.diag([10, 10, 5, 1])  # 状态误差权重
R = np.diag([0.1])  # 控制输入权重

A = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[0.1], [0.1], [1], [0.1]])
X_k = np.array([2, 1, 0.5, 2])  # 当前位置
C = np.array([0, 0, 0, 0])  # 偏移项

# ==== 2. 计算 H 和 F ====
H = K.T @ Q @ K + R
F = 2 * K.T @ Q @ (A @ X_k + C)

# ==== 3. QP 求解 ====
U = cp.Variable((1, 1))  # 控制输入变量
cost = 0.5 * cp.quad_form(U, H) + F @ U
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

print("最优控制输入 U:", U.value)
