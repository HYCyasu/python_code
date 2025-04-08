# Import numpy and OpenCV
import numpy as np
import cv2
#import out

cap = cv2.VideoCapture('video_seq_1.avi')  #video_out2.avi 改成 video_out1.avi

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编解码器
# 视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置输出视频的格式
out = cv2.VideoWriter('video_out2.avi', fourcc, fps, (w, h))   #video_out2.avi 改成 video_out1.avi

# 读取第一帧
_, prev = cap.read()
# Convert frame to grayscale 将其转化为灰度图
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
transforms = np.zeros((n_frames-1, 3), np.float32)
#流光跟踪
for i in range(n_frames-2):
# 在前一帧中检测特征点
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)

  # 读取下一帧
  success, curr = cap.read()
  if not success:
    break
  # 将当前帧转换为灰度图像
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
  # 计算光流
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
  assert prev_pts.shape == curr_pts.shape
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]
  m,n = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
  dx = m[0,2]
  dy = m[1,2]
  da = np.arctan2(m[1,0], m[0,0])
  transforms[i] = [dx,dy,da]
  prev_gray = curr_gray
  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# 定义一个函数，应用移动平均滤波器对变换曲线进行平滑
def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  f = np.ones(window_size)/window_size
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  curve_smoothed = curve_smoothed[radius:-radius]
# return smoothed curve
  return curve_smoothed
def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=60)#30
  return smoothed_trajectory

trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory);
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference
# 定义一个函数，修复仿射变换后可能出现的边界问题
def fixBorder(frame):
  s = frame.shape
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(n_frames-2):
  success, frame = cap.read()
  if not success:
    break

  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]

  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
  frame_stabilized = fixBorder(frame_stabilized)
  frame_out = cv2.hconcat([frame, frame_stabilized])

#显示
  cv2.imshow("Before and After", frame_out)
#等待50毫秒后继续处理下一帧
  cv2.waitKey(50)
#保存
  out.write(frame_stabilized)

