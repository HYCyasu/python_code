import cv2
import numpy as np


def stabilize_video(input_path, output_filename, use_homography=False):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频:", input_path)
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造输出文件的完整路径，确保保存在工作目录中
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 读取第一帧作为初始参考
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧:", input_path)
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 初始化累计变换矩阵（3x3单位矩阵）
    cumulative_transform = np.eye(3)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 检测上一帧的角点
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is None:
            prev_pts = np.array([])

        # 计算光流，跟踪角点
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # 选择跟踪成功的点
        if prev_pts is not None and curr_pts is not None:
            idx = np.where(status.flatten() == 1)[0]
            prev_good = prev_pts[idx]
            curr_good = curr_pts[idx]
        else:
            prev_good = np.empty((0, 1, 2), dtype=np.float32)
            curr_good = np.empty((0, 1, 2), dtype=np.float32)

        if len(prev_good) < 4:  # 如果匹配点太少，则跳过校正
            # 直接使用原图
            stabilized_frame = curr_frame
        else:
            if use_homography:
                # 使用透视变换估计（至少需要4个匹配点）
                H, status_h = cv2.findHomography(prev_good, curr_good, cv2.RANSAC, 5.0)
                if H is None:
                    H = np.eye(3)
                cumulative_transform = cumulative_transform @ np.linalg.inv(H)
                stabilized_frame = cv2.warpPerspective(curr_frame, cumulative_transform, (width, height))
            else:
                # 使用仿射变换估计（仅考虑旋转、平移和尺度）
                affine, inliers = cv2.estimateAffinePartial2D(prev_good, curr_good, method=cv2.RANSAC)
                if affine is None:
                    affine = np.eye(2, 3, dtype=np.float32)
                # 将2x3仿射矩阵扩展为3x3矩阵
                affine_3x3 = np.vstack([affine, [0, 0, 1]])
                cumulative_transform = cumulative_transform @ np.linalg.inv(affine_3x3)
                stabilized_frame = cv2.warpAffine(curr_frame, cumulative_transform[0:2, :], (width, height))

        # 将校正后的帧写入输出文件
        out.write(stabilized_frame)

        # 实时显示（可选）
        cv2.imshow('Stabilized Frame', stabilized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 更新上一帧数据
        prev_gray = curr_gray.copy()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频处理完毕，输出文件：", output_filename)


if __name__ == "__main__":
    # 视频1：静态场景
    video1 = "video_seq_1.avi"
    # 使用仿射变换处理
    stabilize_video(video1, "stabilized_video_seq_1_affine.avi", use_homography=False)
    # 使用透视变换处理
    stabilize_video(video1, "stabilized_video_seq_1_homography.avi", use_homography=True)

    # 视频2：包含移动物体的场景
    video2 = "video_seq_2.avi"
    # 使用仿射变换处理
    stabilize_video(video2, "stabilized_video_seq_2_affine.avi", use_homography=False)
    # 使用透视变换处理
    stabilize_video(video2, "stabilized_video_seq_2_homography.avi", use_homography=True)
