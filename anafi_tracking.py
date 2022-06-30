from pynput import keyboard
import time
import cv2

import csv
import math
import os
# import queue
import shlex
import subprocess
import tempfile
import threading
from olympe.messages.camera import (
    set_camera_mode,
    set_streaming_mode,
)

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged


olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

def main():
    streaming = Streaming()
    # Start the video stream
    streaming.start()
    while not streaming.finish : 
        pass
    # Stop the video stream
    streaming.stop()


class Streaming:
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        self.drone(set_streaming_mode(cam_id=0, value=0))

        # self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
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
        self.yuv_frame_dict = {}

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
        self.running = True
        self.processing_thread.start()

    def stop(self):
        self.running = False
        self.processing_thread.join()

        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.
            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()

        self.yuv_frame_dict.clear()
        self.yuv_frame_dict['frame'] = yuv_frame
        
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
