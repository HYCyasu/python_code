def traditional_afsa(env, start, goal,
                     fish_count=20,
                     initial_vision=5, min_vision=1,
                     initial_step=1, max_iter=100, try_num=5):
    # 初始化鱼群
    population = init_population(env, start, goal, fish_count)
    best_path = min(population, key=path_length)

    for iteration in range(1, max_iter+1):
        # 计算衰减因子（原 α）
        decay_factor = math.exp(-20 * iteration / max_iter)

        # 计算当前视野范围（原 Vis）
        vision_range = min_vision + (initial_vision - min_vision) * decay_factor

        # 计算当前步长（原 Step）
        step_size = initial_step * decay_factor + 0.3

        # 计算收敛误差（用于后面可选的拥挤因子）
        error = abs(path_length(best_path) - path_length(min(population, key=path_length)))

        # 计算拥挤因子（原 δ）
        if error >= 0.03:
            crowding_factor = math.tanh(1.0 / error)
        else:
            crowding_factor = 1.0

        new_population = []
        for path in population:
            # 觅食行为：在 step_size 范围内随机生成若干“邻域”路径
            neighbors = []
            for _ in range(try_num):
                if len(path) > 2:
                    cut_index = random.randrange(1, len(path)-1)
                    new_tail = random_path(env, path[cut_index], goal, max_steps=int(step_size*10))
                    if new_tail:
                        neighbors.append(path[:cut_index] + new_tail)

            # 如果有更短的邻域路径，就移动到最短的那条
            if neighbors:
                best_neighbor = min(neighbors, key=path_length)
                if path_length(best_neighbor) < path_length(path):
                    path = best_neighbor

            new_population.append(path)

        # 更新全局最优
        current_best = min(new_population, key=path_length)
        if path_length(current_best) < path_length(best_path):
            best_path = current_best

        population = new_population

        # 每 10 次迭代打印一次
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{max_iter}, best length = {path_length(best_path):.2f}")

    return best_path
