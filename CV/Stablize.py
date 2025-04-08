import cv2
import numpy as np

class SmoothVideoProcessor:
    """
    该类用于实现视频的稳定化处理，通过光流跟踪计算帧间变换，
    支持仿射和透视两种变换模式。
    """

    def __init__(self, src_filename: str, dst_filename: str, enable_perspective: bool = False):
        """
        初始化视频处理对象。
        :param src_filename: 源视频文件路径。
        :param dst_filename: 输出视频文件路径。
        :param enable_perspective: 是否启用透视变换，False时采用仿射变换。
        """
        self.src_filename = src_filename
        self.dst_filename = dst_filename
        self.enable_perspective = enable_perspective

        self.frame_rate = None
        self.frame_width = None
        self.frame_height = None

        self.vid_capture = None
        self.vid_writer = None

        # 累积变换矩阵，初始为单位矩阵（3x3）
        self.accumulated_matrix = np.eye(3, dtype=np.float32)

    def initialize_io(self):
        """打开视频文件，并初始化视频写入器。"""
        self.vid_capture = cv2.VideoCapture(self.src_filename)
        if not self.vid_capture.isOpened():
            print("无法打开视频:", self.src_filename)
            return

        self.frame_rate = self.vid_capture.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.vid_writer = cv2.VideoWriter(self.dst_filename, fourcc, self.frame_rate, (self.frame_width, self.frame_height))

    def compute_transform(self, gray_prev_frame, gray_curr_frame, curr_color_frame):
        """
        通过检测与跟踪角点计算当前帧的稳定化变换，并返回校正后的帧。
        :param gray_prev_frame: 前一帧的灰度图像。
        :param gray_curr_frame: 当前帧的灰度图像。
        :param curr_color_frame: 当前帧的彩色图像。
        :return: 校正后的当前帧
        """
        prev_features = cv2.goodFeaturesToTrack(gray_prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_features is None:
            prev_features = np.empty((0, 1, 2), dtype=np.float32)

        curr_features, flow_mask, _ = cv2.calcOpticalFlowPyrLK(gray_prev_frame, gray_curr_frame, prev_features, None)
        if prev_features is not None and curr_features is not None:
            valid_idx = np.where(flow_mask.flatten() == 1)[0]
            valid_prev_points = prev_features[valid_idx]
            valid_curr_points = curr_features[valid_idx]
        else:
            valid_prev_points = np.empty((0, 1, 2), dtype=np.float32)
            valid_curr_points = np.empty((0, 1, 2), dtype=np.float32)

        if len(valid_prev_points) < 4:
            return curr_color_frame

        if self.enable_perspective:
            # 透视变换（需要至少4个匹配点）
            trans_matrix, _ = cv2.findHomography(valid_prev_points, valid_curr_points, cv2.RANSAC, 5.0)
            if trans_matrix is None:
                trans_matrix = np.eye(3, dtype=np.float32)
            self.accumulated_matrix = self.accumulated_matrix @ np.linalg.inv(trans_matrix)
            fixed_frame = cv2.warpPerspective(curr_color_frame, self.accumulated_matrix, (self.frame_width, self.frame_height))
        else:
            # 仿射变换
            affine_trans, _ = cv2.estimateAffinePartial2D(valid_prev_points, valid_curr_points, method=cv2.RANSAC)
            if affine_trans is None:
                affine_trans = np.eye(2, 3, dtype=np.float32)
            affine_3x3 = np.vstack([affine_trans, [0, 0, 1]])
            self.accumulated_matrix = self.accumulated_matrix @ np.linalg.inv(affine_3x3)
            fixed_frame = cv2.warpAffine(curr_color_frame, self.accumulated_matrix[0:2, :], (self.frame_width, self.frame_height))

        return fixed_frame

    def process_frames(self):
        """逐帧处理视频，并将稳定化后的帧写入输出，同时实时显示处理结果。"""
        ret, first_frame = self.vid_capture.read()
        if not ret:
            print("无法读取视频第一帧:", self.src_filename)
            return

        prev_gray_image = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = self.vid_capture.read()
            if not ret:
                break

            curr_gray_image = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            fixed_frame = self.compute_transform(prev_gray_image, curr_gray_image, curr_frame)
            self.vid_writer.write(fixed_frame)
            cv2.imshow("稳定后的视频帧", fixed_frame)
            cv2.waitKey(1)  # 仅用于刷新预览，不检测退出按键
            prev_gray_image = curr_gray_image.copy()

    def cleanup(self):
        """释放视频捕获与写入资源，并关闭所有窗口。"""
        if self.vid_capture is not None:
            self.vid_capture.release()
        if self.vid_writer is not None:
            self.vid_writer.release()
        cv2.destroyAllWindows()

    def execute(self):
        """执行整个视频稳定化处理流程。"""
        self.initialize_io()
        self.process_frames()
        self.cleanup()
        print("视频稳定化完成，输出文件为:", self.dst_filename)


def smooth_video(src_filename: str, dst_filename: str, enable_perspective: bool = False):
    processor = SmoothVideoProcessor(src_filename, dst_filename, enable_perspective)
    processor.execute()


if __name__ == "__main__":

    video_file1 = "video_seq_1.avi"
    smooth_video(video_file1, "smoothed_video_seq_1_affine.avi", enable_perspective=False)
    smooth_video(video_file1, "smoothed_video_seq_1_perspective.avi", enable_perspective=True)

    video_file2 = "video_seq_2.avi"
    smooth_video(video_file2, "smoothed_video_seq_2_affine.avi", enable_perspective=False)
    smooth_video(video_file2, "smoothed_video_seq_2_perspective.avi", enable_perspective=True)
