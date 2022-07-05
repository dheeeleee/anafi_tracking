from pynput import keyboard
import time
import cv2
from hsv_detection import HSV

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh
from models.common import DetectMultiBackend

from utils.torch_utils import select_device, time_sync

import csv
import math
import os
# import queue
import shlex
import subprocess
import tempfile
import threading
# from olympe.messages.camera import (
#     set_camera_mode,
#     set_streaming_mode,
# )

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged


olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

# WEIGHTS = '/home/aims/drone_640_s.pt'
WEIGHTS = '/home/aims/drone_1280_s.pt'
IMG_SIZE = 640
DEVICE = '0'
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

def yolo(image, model, AUGMENT = False) : 
    x , y, w, h, conf = 0, 0, 0, 0, 0.
        # Load image
    frame = image # BGR
    xyxy = [0, 0, 0, 0]

    # Padded resize
    img = letterbox(frame, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t0 = time_sync()
    pred = model(img, augment=AUGMENT)[0]
    print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]
    print('det shape:', det.shape)

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    
    original = [0, 0, 0, 0]

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            xyxy = np.array(torch.tensor(xyxy).view(1, 4).view(-1).tolist())
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
            x=xywh[0]
            y=xywh[1]
            w=xywh[2]
            h=xywh[3]
            print(xyxy)
            print(conf)
            
        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

    # Stream results
    print(s)
    # frame = cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])) ,(int(xyxy[2]),int(xyxy[3])) , (0,255,0),thickness = 5)
    # frame = cv2.putText(frame, "x : %d   y : %d  w : %d  h : %d  conf : %.2f " %(x, y, w, h, conf), (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)

    return xyxy, x, y, w, h, conf

def main():
    streaming = Streaming()
    # Start the video stream
    streaming.start()
    while not streaming.finish : 
        try : 
            frame = streaming.frame_dict['frame']
        except KeyError : 
            continue
        xyxy, x, y, w, h, conf = yolo(frame, model)

        frame_disp = frame.copy()
        cv2.rectangle(frame_disp, (int(xyxy[0]),int(xyxy[1])) ,(int(xyxy[2]),int(xyxy[3])) , (0,255,0),thickness = 5)
        cv2.putText(frame_disp, "x : %d   y : %d  w : %d  h : %d  conf : %.2f " %(x, y, w, h, conf), (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('result',frame_disp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    # Stop the video stream
    streaming.stop()


class Streaming:
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        # self.drone(set_streaming_mode(cam_id=0, value=0))

        # self.frame_queue = queue.Queue()
        # self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.tracking = False
        self.angle = 40
        self.yaw = 40
        self.gaz = 10
        self.input = {
            'pitch' : 0,
            'roll' : 0,
            'yaw' : 0,
            'gaz' : 0,
        }
        self.finish = False
        self.pressed = set()
        self.init_controls()
        # self.yuv_frame_dict = {}
        self.frame_dict = {}

        # self.purple_lower = (113, 75, 14)
        # self.purple_upper = (173, 255, 255)
        # self.hsv = HSV(720, 1280, self.purple_lower, self.purple_upper)

    def on_press(self, keyname):
        """handler for keyboard listener"""
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)

            self.pressed.add(keyname)

            if keyname == 'Key.esc':
                self.drone(Landing())
                self.finish = not self.finish
                exit(0)
            if keyname == 't':
                self.tracking = not self.tracking
                print('Tracking : ', self.tracking)
            if keyname == 'Key.tab' : 
                self.drone(TakeOff())
            if keyname == 'Key.backspace' : 
                self.drone(Landing())

            if keyname in self.controls:
                key_handler = self.controls[keyname][0]

                if isinstance(key_handler, str):
                    self.input[key_handler] = int(self.controls[keyname][1])
                else:
                    key_handler()
            self.drone(PCMD(1, self.input['roll'], self.input['pitch'], self.input['yaw'], self.input['gaz'], timestampAndSeqNum=0))

        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""

        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.pressed : 
            self.pressed.remove(keyname)

            if keyname in self.controls:
                key_handler = self.controls[keyname][0]

                if isinstance(key_handler, str):
                    self.input[key_handler] = int(0)
                else:
                    key_handler()
            self.drone(PCMD(1, self.input['roll'], self.input['pitch'], self.input['yaw'], self.input['gaz'], timestampAndSeqNum=0))

    def init_controls(self):
        """Define keys and add listener"""
        self.controls = {
            'w': ['gaz', str(self.gaz)] ,
            's': ['gaz', str(-self.gaz)],
            'a': ['yaw', str(-self.yaw)],
            'd': ['yaw', str(self.yaw)],
            'Key.left': ['roll', str(-self.angle)],
            'Key.right': ['roll', str(self.angle)],
            'Key.up': ['pitch', str(self.angle)],
            'Key.down': ['pitch', str(-self.angle)],
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

    def start(self):
        # Connect the the drone
        assert self.drone.connect(retry=3)

        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"



        # Setup your callback functions to do some live video processing
        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
        )
        # Start video streaming
        self.drone.streaming.start()
        # self.running = True
        # self.processing_thread.start()

    def stop(self):
        self.running = False
        # self.processing_thread.join()

        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.
            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()

        # self.yuv_frame_dict.clear()
        # self.yuv_frame_dict['frame'] = yuv_frame
        cv2_cvt_color_flag = {
                olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
            }[yuv_frame.format()]

        self.frame_dict.clear()
        self.frame_dict['frame'] = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)

        yuv_frame.unref()
        
        # using queue
        # with self.frame_queue.mutex : 
        #     self.frame_queue.queue.clear()

        # self.frame_queue.put_nowait(yuv_frame)

    def yuv_frame_processing(self):
        a = 0
        while self.running:
            # using queue
            # try:
            #     # yuv_frame = self.frame_queue.get(timeout=0.1)
            #     yuv_frame = self.frame_queue.get()

            #     with self.frame_queue.mutex : 
            #         self.frame_queue.queue.clear()

            # except queue.Empty:
            #     continue
            try : 
                yuv_frame = self.yuv_frame_dict['frame']
                self.yuv_frame_dict.clear()
                
            except KeyError : 
                continue

            # You should process your frames here and release (unref) them when you're done.
            # Don't hold a reference on your frames for too long to avoid memory leaks and/or memory
            # pool exhaustion.

            info = yuv_frame.info()

            height, width = (  # noqa
                info["raw"]["frame"]["info"]["height"],
                info["raw"]["frame"]["info"]["width"],
            )

            cv2_cvt_color_flag = {
                olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
            }[yuv_frame.format()]

            self.cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)

            if self.tracking : 
                pass

            cv2.imshow('anafi', self.cv2frame)
            cv2.waitKey(1)

            yuv_frame.unref()

if __name__ == "__main__":
    main()
