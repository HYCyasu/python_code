import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# 读入图像（请确保文件路径正确）
img = cv2.imread('img.png')
if img is None:
    print("无法加载图像，请检查文件路径。")
    exit()

# 将图像转换到 HSV 色彩空间，有利于颜色分割
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 假设这四条曲线分别代表：电负荷、热负荷、风机、光伏
# 这里给出一个示例 HSV 范围。你需要根据图中实际颜色做调整。
color_ranges = {
    '电负荷': {'lower': np.array([0, 70, 50]), 'upper': np.array([10, 255, 255])},  # 示例：红色部分
    '热负荷': {'lower': np.array([100, 150, 0]), 'upper': np.array([140, 255, 255])},  # 示例：蓝色部分
    '风机': {'lower': np.array([40, 70, 50]), 'upper': np.array([80, 255, 255])},  # 示例：绿色部分
    '光伏': {'lower': np.array([10, 100, 100]), 'upper': np.array([25, 255, 255])}  # 示例：橙色部分
}

# ----- 第一步：手动校准图像坐标系 -----
# 根据图像中坐标轴刻度确定以下变量（示例数值，仅供参考）：
# 假设 x 轴像素 [x_min, x_max] 对应实际时间 [0, 24] 小时
# 假设 y 轴像素 [y_max, y_min]（注意图像坐标原点在左上角）对应实际数值 [y_value_min, y_value_max]
x_min, x_max = 100, 800  # 例如：x=100 像素对应 0 小时，x=800 像素对应 24 小时
y_max, y_min = 550, 100  # 例如：y=550 像素对应数值下限（0），y=100 像素对应上限（100）
time_start, time_end = 0, 24
y_value_min, y_value_max = 0, 100  # 根据图像调整实际数值范围


def pixel_to_time(x_pixel):
    """将 x 像素位置映射到实际时间（小时）"""
    return (x_pixel - x_min) / (x_max - x_min) * (time_end - time_start) + time_start


def pixel_to_value(y_pixel):
    """将 y 像素位置映射到实际数据值
       注意：图像中 y 越大表示越低，因此需要反转映射关系
    """
    return (y_max - y_pixel) / (y_max - y_min) * (y_value_max - y_value_min) + y_value_min


# 用于存储每条曲线采样后的数据
curve_data = {}

# ----- 第二步：对每条曲线进行提取与采样 -----
for label, ranges in color_ranges.items():
    # 根据颜色范围生成掩码
    mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
    # 对掩码进行形态学处理以减少噪声
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # 查找曲线轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"未找到 {label} 曲线")
        continue

    # 选择面积最大的轮廓作为目标
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()  # 去除多余的维度

    # 按 x 坐标排序，确保从左到右的顺序
    sorted_idx = np.argsort(contour[:, 0])
    contour_sorted = contour[sorted_idx]

    # 提取 x 与 y 像素坐标
    x_pixels = contour_sorted[:, 0]
    y_pixels = contour_sorted[:, 1]

    # 将像素坐标转换为实际的时间和数值
    times = np.array([pixel_to_time(x) for x in x_pixels])
    values = np.array([pixel_to_value(y) for y in y_pixels])

    # 对数据进行插值，使得在整个时间区间上连续，并以一个小时为步长采样
    try:
        f_interp = interp1d(times, values, kind='linear', bounds_error=False, fill_value="extrapolate")
    except Exception as e:
        print(f"{label} 插值出错：", e)
        continue
    times_hourly = np.arange(time_start, time_end + 1, 1)  # 每小时一步
    values_hourly = f_interp(times_hourly)

    curve_data[label] = (times_hourly, values_hourly)

    # 绘制检测结果供参考验证
    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x_pixels, y_pixels, 'k-', lw=2, label='提取轮廓')
    # 由实际采样值反向映射到像素坐标便于验证
    x_pixels_sampled = ((times_hourly - time_start) / (time_end - time_start)) * (x_max - x_min) + x_min
    y_pixels_sampled = y_max - (values_hourly - y_value_min) * (y_max - y_min) / (y_value_max - y_value_min)
    plt.plot(x_pixels_sampled, y_pixels_sampled, 'ro', label='采样点')
    plt.title(f"{label} 曲线数据提取")
    plt.legend()
    plt.show()

# ----- 第三步：输出与保存数据 -----
for label, (times_hourly, values_hourly) in curve_data.items():
    print(f"--- {label} ---")
    for t, v in zip(times_hourly, values_hourly):
        print(f"时间: {t:.1f} 小时, 数值: {v:.2f}")

# 整合各曲线数据后保存到 CSV 文件中
dfs = []
for label, (times_hourly, values_hourly) in curve_data.items():
    df = pd.DataFrame({'时间(小时)': times_hourly, label: values_hourly})
    df.set_index('时间(小时)', inplace=True)
    dfs.append(df)
df_all = pd.concat(dfs, axis=1)
df_all.to_csv("extracted_curve_data.csv")
print("所有曲线数据已保存至 extracted_curve_data.csv")
