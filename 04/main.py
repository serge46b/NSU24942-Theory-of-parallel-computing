from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
import time
import logging
from threading import Event, Thread
import argparse
from typing import Callable

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")


class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self):
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    """Sensor Cam"""

    video_capturer: cv2.VideoCapture

    def __init__(self, cam_name: str, cam_res: str):
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

    def get(self) -> cv2.typing.MatLike | None:
        ret, frame = self.video_capturer.read()
        if not ret:
            logging.error("(SensorCam): Unable to read frame")
            return
        return frame


class WindowImage:
    def __init__(self, fps: int):
        self.fps = fps
        self.window_name = "Result"
        cv2.namedWindow(self.window_name)

    def __del__(self):
        cv2.destroyWindow(self.window_name)

    def show(self, data: cv2.typing.MatLike) -> int:
        cv2.imshow(self.window_name, data)
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


def sensor_worker(sensor: Sensor, queue: Queue, stop_event: Event) -> None:
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


@dataclass
class SensorSData:
    sensor_worker: Callable[[Sensor, Queue, Event], None]
    sensor_queue: Queue
    data: None | cv2.typing.MatLike | int
    thread: Thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-name", type=str, default="0")
    parser.add_argument("--cam-res", type=str, default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    setup_logging()
    logging.info("(main): Starting program")
    sensor0 = SensorX(1)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(0.01)
    cam_sensor = SensorCam(args.cam_name, args.cam_res)
    window = WindowImage(args.fps)
    stop_event = Event()
    sensor_sdata: list[SensorSData] = []
    for sensor in [sensor0, sensor1, sensor2, cam_sensor]:
        queue = Queue()
        worker = sensor_worker
        thread = Thread(target=worker, args=(sensor, queue, stop_event))
        thread.start()
        sensor_sdata.append(SensorSData(worker, queue, None, thread))
    l_data: list[None | cv2.typing.MatLike | int] = [None] * len(sensor_sdata)
    l_frame: cv2.typing.MatLike | None = None
    while not stop_event.is_set():
        try:
            frame = get_latest_data(sensor_sdata[3].sensor_queue)
            if frame is not None:
                l_frame = frame
            if l_frame is None:
                continue
            w_frame = l_frame.copy()
            for i in range(len(sensor_sdata) - 1):
                data = get_latest_data(sensor_sdata[i].sensor_queue)
                if data is not None:
                    l_data[i] = data
                cv2.putText(
                    w_frame,
                    f"Sensor {i}: {l_data[i]}",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            q = window.show(w_frame)
            if q == 27:
                stop_event.set()
        except KeyboardInterrupt:
            logging.info("(main): Keyboard interrupt")
            stop_event.set()
        except Exception as e:
            logging.error(f"(main): Error in loop: {e}")
            stop_event.set()
    for sdata in sensor_sdata:
        sdata.thread.join()
    logging.info("(main): Stopped")
