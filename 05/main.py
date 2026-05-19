from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
import time
import logging
from threading import Event, Thread
import argparse
from typing import Callable

import cv2
from ultralytics.models.yolo import YOLO
import torch


@dataclass
class FrameData:
    frame_idx: int
    frame: cv2.typing.MatLike


class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")


class Predictor:
    def __init__(self, model_path: str):
        print(f"Loading {model_path}")
        self.model = YOLO(model_path)
        print(f"Model loaded")

    def predict(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike | None:
        try:
            return self.model.predict(frame, verbose=False)[0].plot(
                line_width=2, kpt_radius=5, boxes=True
            )
        except Exception as e:
            logging.error(f"(Predictor): Error in predict: {e}")
            return None


class SensorCam(Sensor):
    """Sensor Cam"""

    video_capturer: cv2.VideoCapture

    def __init__(self, cam_name: str, cam_res: str):
        self.frame_idx = 0
        conv_cam_name = int(cam_name) if cam_name.isnumeric() else cam_name
        try:
            cam_w, cam_h = map(int, cam_res.lower().split("x"))
        except ValueError:
            logging.error(
                f"(SensorCam) Parsing error: resolution must be in WxH format, got: {cam_res}"
            )
            return
        self.video_capturer = cv2.VideoCapture(conv_cam_name)
        if not self.video_capturer.isOpened():
            logging.error(f"(SensorCam): Unable to open camera '{cam_name}'")
            return
        self.video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    def __del__(self):
        self.video_capturer.release()

    def get(self) -> FrameData | None:
        self.frame_idx += 1
        ret, frame = self.video_capturer.read()
        if not ret:
            logging.error("(SensorCam): Unable to read frame")
            return
        return FrameData(frame_idx=self.frame_idx, frame=frame)


class WindowImage:
    def __init__(self, fps: int, name: str = "Result"):
        self.fps = fps
        self.window_name = name
        cv2.namedWindow(self.window_name)

    def __del__(self):
        cv2.destroyWindow(self.window_name)

    def show(self, data: cv2.typing.MatLike, nowait: bool = False) -> int:
        cv2.imshow(self.window_name, data)
        if nowait:
            return 0
        return cv2.waitKey(int(1000 / self.fps))


def setup_logging() -> None:
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "main.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def put_latest_data(queue: Queue, data) -> None:
    try:
        queue.put_nowait(data)
    except Full:
        queue.get_nowait()
        queue.put_nowait(data)


def get_latest_data(queue: Queue):
    data = None
    while not queue.empty():
        data = queue.get_nowait()
    return data


def sensor_worker(sensor: Sensor, queue: Queue[FrameData], stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            data = sensor.get()
            if data is not None:
                put_latest_data(queue, data)
            else:
                logging.error("(sensor_worker): Unable to read frame")
                stop_event.set()
        except Exception as e:
            logging.error(f"(sensor_worker): Error in sensor worker: {e}")
            stop_event.set()
    logging.info("(sensor_worker): stopped")


def predictor_worker(
    predictor: Predictor,
    input_queue: Queue[FrameData],
    output_queue: Queue[FrameData],
    stop_event: Event,
) -> None:
    while not stop_event.is_set():
        try:
            frame_data = get_latest_data(input_queue)
            if frame_data is None:
                continue
            result = predictor.predict(frame_data.frame)
            if result is not None:
                put_latest_data(
                    output_queue,
                    FrameData(frame_idx=frame_data.frame_idx, frame=result),
                )
        except Exception as e:
            logging.error(f"(predictor_worker): Error in predictor worker: {e}")
            stop_event.set()
    logging.info("(predictor_worker): stopped")


def insert_processing_frames(
    processed_frames: list[FrameData], frame_data: FrameData
) -> None:
    left, right = 0, len(processed_frames)
    while left < right:
        mid = (left + right) // 2
        if processed_frames[mid].frame_idx < frame_data.frame_idx:
            left = mid + 1
        else:
            right = mid
    processed_frames.insert(left, frame_data)


def main():
    import os

    # Limit PyTorch / NumPy multithreading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-name", type=str, default="0")
    parser.add_argument("--cam-res", type=str, default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    logging.info("(main): Starting program")
    sensor = SensorCam(args.cam_name, args.cam_res)
    window = WindowImage(args.fps, "Original")
    window_predict = WindowImage(args.fps, "Predicted")
    stop_event = Event()
    sensor_queue = Queue()
    sensor_thread = Thread(
        target=sensor_worker, args=(sensor, sensor_queue, stop_event)
    )
    sensor_thread.start()
    predictor_threads = []
    predictor_in_queues = []
    predictor_out_queues = []
    for i in range(args.workers):
        predictor_out_queue = Queue()
        predictor_in_queue = Queue()
        predictor = Predictor("yolov8n-pose.pt")
        predictor_thread = Thread(
            target=predictor_worker,
            args=(predictor, predictor_in_queue, predictor_out_queue, stop_event),
        )
        predictor_thread.start()
        predictor_threads.append(predictor_thread)
        predictor_out_queues.append(predictor_out_queue)
        predictor_in_queues.append(predictor_in_queue)

    processed_frames: list[FrameData] = []
    last_pushed_worker_idx = 0
    l_frame_update_time = time.time()

    while not stop_event.is_set():
        try:
            frame_data = get_latest_data(sensor_queue)
            if frame_data is None:
                continue
            put_latest_data(predictor_in_queues[last_pushed_worker_idx], frame_data)
            last_pushed_worker_idx = (last_pushed_worker_idx + 1) % args.workers
            for i in range(args.workers):
                frame_pred_data = get_latest_data(predictor_out_queues[i])
                if frame_pred_data is not None:
                    # print(f"Worker i: {i}")
                    insert_processing_frames(processed_frames, frame_pred_data)
            if len(processed_frames) > 0:
                # print(f"Processed frames: {len(processed_frames)}")
                true_fps = 1 / (time.time() - l_frame_update_time)
                cv2.putText(
                    processed_frames[0].frame,
                    f"True FPS: {true_fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                window_predict.show(processed_frames[0].frame, nowait=True)
                l_frame_update_time = time.time()
                processed_frames.pop(0)
            q = window.show(frame_data.frame)
            if q == 27:
                stop_event.set()
        except KeyboardInterrupt:
            logging.info("(main): Keyboard interrupt")
            stop_event.set()
        except Exception as e:
            logging.error(f"(main): Error in loop: {e}")
            stop_event.set()


if __name__ == "__main__":
    main()
