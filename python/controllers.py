"""
Implementation of controllers for simulated drone (AirSim)
and real drone (DJI Tello EDU).

The simulated drone is implemented twice: a normal and a
noisy version.

Controllers have been implemented to work with the exact same
interface in order to be interchangeable in their use.
"""

###########
# Imports #
###########

import airsim
import cv2
import keyboard
import numpy as np
import socket
import threading
import time

from abc import ABC, abstractmethod
from colored import fg, attr


###########
# Classes #
###########

class Controller(ABC):
    """
    Abstract class used to define a controller.

    Each controller must inherit this class.
    """

    def __init__(self):
        super().__init__()

    # Initialization

    @abstractmethod
    def arm(self):
        pass

    @abstractmethod
    def disarm(self):
        pass

    # Navigation

    @abstractmethod
    def takeoff(self):
        pass

    @abstractmethod
    def land(self):
        pass

    @abstractmethod
    def move(self, direction: str, distance: int, speed: int = 50):
        pass

    @abstractmethod
    def rotate(self, direction: str, angle: int):
        pass

    @abstractmethod
    def hover(self):
        pass

    # Camera

    @abstractmethod
    def picture(self) -> np.ndarray:
        pass

    # Common

    def manual(self):
        while True:
            if keyboard.is_pressed('up'):
                self.move('up', 20)
            elif keyboard.is_pressed('down'):
                self.move('down', 20)
            elif keyboard.is_pressed('left'):
                self.move('left', 20)
            elif keyboard.is_pressed('right'):
                self.move('right', 20)
            elif keyboard.is_pressed('z'):
                self.move('forward', 20)
            elif keyboard.is_pressed('s'):
                self.move('back', 20)
            elif keyboard.is_pressed('q'):
                self.rotate('ccw', 5)
            elif keyboard.is_pressed('d'):
                self.rotate('cw', 5)
            elif keyboard.is_pressed('c'):
                break


class AirSimDrone(Controller):
    """
    Implementation of the controller used for the simulated
    drone in AirSim.

    This version uses the functions of AirSim API without
    any noise.
    """

    def __init__(self):
        super().__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    # Abstract interface

    def arm(self):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def disarm(self):
        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)

    def takeoff(self):
        landed = self.state().landed_state

        if landed == airsim.LandedState.Landed:
            self.client.takeoffAsync().join()
            self.move('down', 80, 50)

    def land(self):
        landed = self.state().landed_state

        if landed != airsim.LandedState.Landed:
            self.client.landAsync().join()

    def move(self, direction, distance, speed=50):
        error = False

        if speed < 10 or speed > 100:
            print('Speed must be between 10 and 100 [cm/s]')

            error = True

        if distance < 20 or distance > 500:
            print('Distance must be between 20 and 500 [cm]')

            error = True

        speed /= 100
        duration = (distance / 100) / speed

        directions = {
            'up': [0, 0, -speed, duration],
            'down': [0, 0, speed, duration],
            'left': [0, -speed, 0, duration],
            'right': [0, speed, 0, duration],
            'forward': [speed, 0, 0, duration],
            'back': [-speed, 0, 0, duration]
        }

        factors = directions.get(direction)

        if factors is None:
            print('Unknown direction')

            error = True

        if not error:
            self.client.moveByVelocityBodyFrameAsync(*factors).join()

    def rotate(self, direction, angle):
        error = False

        factors = {
            'cw': 1,
            'ccw': -1
        }

        factor = factors.get(direction)

        if factor is None:
            print('Unknown direction')

            error = True

        if angle < 1 or angle > 360:
            print('Angle must be between 1 and 360 [degree]')

            error = True

        rate = factor * 45
        duration = angle / 45

        if not error:
            self.client.rotateByYawRateAsync(rate, duration).join()

    def hover(self):
        self.client.hoverAsync().join()

    def picture(self):
        pictures = self.client.simGetImages([
            airsim.ImageRequest(
                camera_name='front_center',
                image_type=airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False
            )
        ])
        picture = pictures[0]

        image = airsim.string_to_uint8_array(picture.image_data_uint8)
        image = image.reshape(picture.height, picture.width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    # Specific

    def log(self, message):
        self.client.simPrintLogMessage(message)

    def teleport(self, position):
        pose = self.client.simGetVehiclePose()
        pose.position = airsim.Vector3r(*position)

        self.client.simSetVehiclePose(pose, True)

    def state(self):
        return self.client.getMultirotorState()

    def imu(self):
        return self.client.getImuData()

    def gps(self):
        return self.client.getGpsData()


class AirSimDroneNoisy(AirSimDrone):
    """
    Implementation of the controller used for the simulated
    drone in AirSim.

    This version uses the functions of AirSim API with
    additional noise on movement to simulate the real
    drone behaviour.
    """

    def __init__(self):
        super().__init__()

    def _noise(self, value):
        mean = value / 10
        std = value / 50

        noise = np.random.normal(mean, std)

        return value + noise

    def move(self, direction, distance, speed=50):
        super().move(direction, self._noise(distance), speed)

    def rotate(self, direction, angle):
        super().rotate(direction, self._noise(angle))


class TelloEDU(Controller):
    """
    Implementation of the controller used for the DJI Tello EDU.
    """

    def __init__(self):
        super().__init__()

        # Socket for sending command
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('', 8889))

        # DJI Tello EDU information
        self.tello_address = ('192.168.10.1', 8889)

        # Response of the drone
        self.response = None

        # Time out
        self.max_timeout = 15.0

        # State of the drone
        self.disarmed = True

        # Last frame captured by the camera
        self.frame = None

        # Thread for receiving command response
        self.response_thread = threading.Thread(target=self._response_thread)
        self.response_thread.daemon = True

        # Thread for receiving video stream
        self.stream_thread = threading.Thread(target=self._stream_thread)
        self.stream_thread.daemon = True

    # Abstract interface

    def arm(self):
        self.disarmed = False

        self.response_thread.start()

        self._send_command('command')
        self._send_command('streamon')

        self.stream_thread.start()

    def disarm(self):
        self._send_command('streamoff')

        self.disarmed = True

        self.response_thread.join()
        self.stream_thread.join()

        self.socket.close()

    def takeoff(self):
        self._send_command('takeoff')

    def land(self):
        self._send_command('land')

    def move(self, direction, distance, speed=50):
        error = False

        if speed < 10 or speed > 100:
            print('Speed must be between 10 and 100 [cm/s]')

            error = True

        if direction not in ['up', 'down', 'left', 'right', 'forward', 'back']:
            print('Unknown direction')

            error = True

        if distance < 20 or distance > 500:
            print('Distance must be between 20 and 500 [cm]')

            error = True

        if not error:
            self._send_command(f'speed {speed}')
            self._send_command(f'{direction} {distance}')

    def rotate(self, direction, angle):
        error = False

        if direction not in ['cw', 'ccw']:
            print('Unknown direction')

            error = True

        if angle < 1 or angle > 360:
            print('Angle must be between 1 and 360 [degree]')

            error = True

        if not error:
            self._send_command(f'{direction} {angle}')

    def hover(self):
        self._send_command('stop')

    def picture(self):
        return self.frame

    # Specific

    def emergency(self):
        self._send_command('emergency')

    def battery(self):
        self._send_command('battery?')

    def _send_command(self, command):
        print(f'Command: {fg(6)}{command}{attr(0)}')

        self.socket.sendto(
            command.encode('utf-8'),
            self.tello_address
        )

        start = time.time()

        while self.response is None:
            now = time.time()

            if now - start > self.max_timeout:
                print('Time out')

                break

        self.response = None

    def _response_thread(self):
        while True:
            if self.disarmed:
                break

            try:
                self.response, ip = self.socket.recvfrom(1024)

                if self.response is not None:
                    self.response = self.response.decode('utf-8').strip()

                    color = fg(2) if self.response == 'ok' else fg(1)

                    print(f'Response: {color}{self.response}{attr(0)}')
            except Exception as e:
                print(f'Exception: {e}')

    def _stream_thread(self):
        stream = cv2.VideoCapture('udp://@0.0.0.0:11111')

        while True:
            if self.disarmed:
                break

            ack, frame = stream.read()

            if ack:
                frame = cv2.resize(frame, (320, 180))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.frame = frame

        stream.release()
