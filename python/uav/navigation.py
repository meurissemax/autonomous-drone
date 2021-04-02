"""
Implementation of navigation modules and algorithms.
"""

###########
# Imports #
###########

import cv2
import math
import numpy as np
import torch
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Union

from analysis.markers import ArucoOpenCV, QROpenCV, QRZBar
from analysis.vanishing_point import VPClassic, VPEdgelets
from .controllers import Controller
from .environment import Environment
from learning.models import DenseNet, SmallConvNet, UNet, MiDaS


##########
# Typing #
##########

Image = np.array


###########
# Classes #
###########

# Modules

# - Abstract class

class NavModule(ABC):
    """
    Abstract class used to define a navigation module.

    A navigation module is a component of a navigation algorithm. It performs
    several actions on drone input(s) and returns some information or directly
    acts on the drone.

    Each navigation module must inherit this class.
    """

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.verbose = verbose

    def log(self, message: str):
        """
        Display a message in console if verbose flag is enabled.
        """

        if self.verbose:
            print(f'[{self.__class__.__name__}] {message}')

    @abstractmethod
    def run(self, **kwargs) -> Union[object, None]:
        """
        Take as inputs some data of the drone (image, IMU data, etc) and return
        some information (action, position, etc) or directly act on the drone.
        """

        pass


# - Analysis

class VanishingModule(NavModule):
    """
    Abstract class used to define navigation module that works with vanishing
    point.
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        self.controller = controller

        # Define adjustment actions
        self.adjustments = {
            'left': partial(self.controller.rotate, 'cw', 5),
            'right': partial(self.controller.rotate, 'ccw', 5)
        }

    @abstractmethod
    def _alignment(self, img: Image) -> str:
        """
        Determine the alignment of the drone (left, right or center) based on
        an image.
        """

        pass

    # Abstract interface of parent class

    def run(self, img: Image):
        while True:
            alignment = self._alignment(img)

            if alignment == 'center':
                break

            self.adjustments.get(alignment)()

            img = self.controller.picture()


class AnalysisVanishingModule(VanishingModule):
    """
    Module that adjusts the drone by estimating its alignment based on a
    vanishing point detection using pure computer vision method.
    """

    def __init__(self, detector_id: str, controller, verbose=False):
        super().__init__(controller, verbose)

        # Vanishing point detector
        detectors = {
            'classic': VPClassic,
            'edgelets': VPEdgelets
        }

        self.detector = detectors.get(detector_id)()

        # Bounds
        self.lower, self.upper = None, None

    # Abstract interface

    def _alignment(self, img):
        # Define bounds, if not defined
        if self.lower is None:
            h, w, _ = img.shape
            mean, std = w / 2, w / 25

            self.lower = (mean - 4 * std, mean - std)
            self.upper = (mean + std, mean + 4 * std)

        # Detect vanishing point
        x, _ = self.detector.detect(img)

        # Get alignment
        alignment = 'center'

        if self.upper[0] < x < self.upper[1]:
            alignment = 'left'
        elif self.lower[0] < x < self.lower[1]:
            alignment = 'right'

        self.log(f'Alignment: {alignment}')

        return alignment


class MarkerModule(NavModule):
    """
    Module that reads the content of a marker, if any, and returns it along
    with coordinates of its corner points.
    """

    def __init__(self, decoder_id: str, verbose=False):
        super().__init__(verbose)

        # Marker decoder
        decoders = {
            'aruco_opencv': ArucoOpenCV,
            'qr_opencv': QROpenCV,
            'qr_zbar': QRZBar
        }

        self.decoder = decoders.get(decoder_id)()

    # Abstract interface

    def run(self, img: Image) -> Tuple[str, List]:
        decoded, pts = self.decoder.decode(img)

        self.log(f'Decoded: {decoded}')

        return decoded, pts


# - Deep Learning

class DeepModule(NavModule):
    """
    Module that returns the output of a neural network using an image from the
    drone as input.
    """

    def __init__(
        self,
        n_channels: int,
        model_id: str,
        weights_pth: str,
        verbose=False
    ):
        super().__init__(verbose)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.device = device

        self.log(f'Device: {device}')

        # Model
        models = {
            'densenet121': partial(DenseNet, densenet_id='121'),
            'densenet161': partial(DenseNet, densenet_id='161'),
            'small': SmallConvNet,
            'unet': UNet,
            'midas': MiDaS
        }

        model = models.get(model_id)(3, n_channels)
        model = model.to(device)

        if model_id != 'midas':
            model.load_state_dict(torch.load(weights_pth, map_location=device))

        model.eval()

        self.model = model

        # GPU warm-up
        if torch.cuda.is_available():
            with torch.no_grad():
                for _ in range(10):
                    if model_id == 'midas':
                        dummy = torch.randn(1, 3, 224, 384)
                    else:
                        dummy = torch.randn(1, 3, 180, 320)

                    _ = self.model(dummy.to(self.device))

        # Processing
        size = (384, 224) if model_id == 'midas' else (320, 180)

        self.preprocess = transforms.Compose([
            lambda x: cv2.resize(x, size),
            transforms.ToTensor()
        ])

        self.to_list = transforms.Compose([
            lambda x: torch.flatten(x),
            lambda x: x.numpy()
        ])

        self.to_img = transforms.Compose([
            transforms.ToPILImage(),
            lambda x: np.array(x)
        ])

    # Abstract interface

    def run(self, img: Image) -> np.array:
        img = self.preprocess(img)

        with torch.no_grad():
            img = img.unsqueeze(0)
            img = img.to(self.device)

            pred = self.model(img).cpu()
            pred = self.to_img(pred) if pred.dim() == 3 else self.to_list(pred)

            self.log(f'Prediction: {pred}')

            return pred


class DeepVanishingModule(VanishingModule):
    """
    Module that adjusts the drone by estimating its alignment based on a
    vanishing point detection using Deep Learning method.
    """

    def __init__(
        self,
        grid_size: int,
        model_id: str,
        weights_pth: str,
        controller,
        verbose=False
    ):
        super().__init__(controller, verbose)

        # Deep module
        self.deep = DeepModule(
            n_channels=grid_size,
            model_id=model_id,
            weights_pth=weights_pth,
            verbose=False
        )

        # Define middle cell
        self.middle = math.floor(grid_size / 2)

    # Abstract interface

    def _alignment(self, img):
        # Get cell that contains vanishing point
        pred = self.deep.run(img)
        cell = np.argmax(pred)

        # Get alignment
        alignment = 'center'

        if cell == self.middle + 1:
            alignment = 'left'
        elif cell == self.middle - 1:
            alignment = 'right'

        self.log(f'Alignment: {alignment}')

        return alignment


class DepthModule(NavModule):
    """
    Module that infers depth map associated to an image and compute distance to
    element in front of the drone.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)

        # Deep module
        self.deep = DeepModule(
            n_channels=2,  # dummy
            model_id='midas',
            weights_pth='',  # dummy
            verbose=False
        )

    # Abstract interface

    def run(self, img: Image) -> float:
        # Get depth map associated to image
        depth = self.deep.run(img=img)

        # Dimensions
        h, w = depth.shape

        # Crop the image to keep center box
        mx, my = w // 2, h // 2
        dx, dy = w // 10, h // 10

        cropped = depth[(mx - dx):(mx + dx), (my - dy):(my + dy)]

        # Compute distance
        distance = np.mean(cropped)

        self.log(f'Distance: {distance}')

        return distance


# - Staircase

class StaircaseModule(NavModule):
    """
    Module that moves the drone in a staircase (up or down) based on an image.
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        self.controller = controller

        # Depth module
        self.depth = DepthModule(verbose=False)

        # Threshold and limit on distance
        self.threshold = 1300
        self.limit = 1800

        # Define staircase actions
        self.sactions = {
            'up': partial(self.controller.move, 'up', 30),
            'down': partial(self.controller.move, 'down', 30),
            'forward': partial(self.controller.move, 'forward', 30)
        }

    # Abstract interface

    def run(self, img: Image, direction: str):
        while True:
            # Get distance to staircase
            distance = self.depth.run(img=img)

            self.log(f'Distance: {distance}')

            # Check limit
            if distance > self.limit:
                break

            # Move the drone, according to distance
            action = direction if distance < self.threshold else 'forward'
            self.sactions.get(action)()

            # Take a new picture
            img = self.controller.picture()


# Navigation algorithms

# - Abstract class

class NavAlgorithm(ABC):
    """
    Abstract class used to define a navigation algorithm.

    A navigation algorithm is an algorithm, based on an environment
    representation and possibly navigation module(s), that moves automatically
    the drone from its initial position to an objective defined.

    Each navigation algorithm must inherit this class.
    """

    def __init__(
        self,
        env: Environment,
        controller: Controller,
        show: bool = False
    ):
        super().__init__()

        self.env = env
        self.controller = controller
        self._show = show

        # Path and associated sequence of actions
        self.path = env.path()
        self.sequence = env.path_to_seq(self.path)

        # Key points of the environment
        self._keypoints = self.env.extract_keypoints(self.path, self.sequence)
        self._keypoints_idx = 0

        # Possible actions of the drone
        self.actions = {
            'forward': lambda d: self.controller.move('forward', d),
            'left': lambda a: self.controller.rotate('ccw', a),
            'right': lambda a: self.controller.rotate('cw', a)
        }

        # Define custom steps for each action
        self.steps = {'forward': 100, 'left': 90, 'right': 90}

        # Common modules used by algorithms
        self.vanishing = AnalysisVanishingModule(
            detector_id='classic',
            controller=self.controller,
            verbose=True
        )

        self.staircase = StaircaseModule(
            controller=self.controller,
            verbose=True
        )

        # Define reference steps and counters for each action (according to
        # environment representation precision) to properly update the agent
        # position in environment representation
        self._refs = {'forward': 100, 'left': 90, 'right': 90}
        self._counters = {'forward': 0, 'left': 0, 'right': 0}

    # Environment representation

    def _env_move(self, action: str, **kwargs):
        """
        Move the agent in the environment representation according to steps
        defined.
        """

        # Update counter
        self._counters[action] += 1

        # Define step and times
        ref = self._refs[action]
        now = self.steps[action]

        step = int(ref / now) if ref > now else 1
        times = 1 if ref > now else int(now / ref)

        # Update agent in environment, if necessary
        if step == 1 or self._counters[action] % step:
            self.env.move(action=action, times=times, **kwargs)

    def _env_update(self, pos: Tuple, **kwargs):
        """
        Update the position of the agent in the environment representation.
        """

        # Reset all counters
        for key in self._counters.keys():
            self._counters[key] = 0

        # Update agent in environment
        self.env.update(pos=pos, **kwargs)

    def _env_show(self, **kwargs):
        """
        Show the environment representation if flag is enabled.
        """

        if self._show:
            self.env.render(**kwargs)

    # Navigation

    @abstractmethod
    def _is_keypoint(self, img: Image) -> bool:
        """
        Define if the drone has reached a key point or not based on an image.

        Each navigation algorithm must implement this method.
        """

        pass

    def _execute(self):
        """
        Define how the drone executes actions during the navigation process.

        By default, it uses images of the drone to determine next action by
        checking if the drone has reach a key point or not.

        This method can be override if the navigation algorithm has to work
        differently.
        """

        while not self.env.has_reached_obj():
            img = self.controller.picture()

            # Align drone using vanishing point
            self.vanishing.run(img=img)

            # Check if drone has reach a key point
            if self._is_keypoint(img):
                action, pos = self._keypoints[self._keypoints_idx]
                self._keypoints_idx += 1

                self._env_update(pos)
            else:
                action = 'forward'

            # Execute action
            if action in ['up', 'down']:
                self.staircase.run(img, action)
            else:
                self.actions.get(action)(self.steps.get(action))
                self._env_move(action)

            # Show updated environment
            self._env_show(path=self.path, what=['pos', 'obj'])

    def navigate(self):
        """
        Move automatically the drone from its initial position to an objective.

        This navigation process follow a general template: the drone takes off,
        executes actions and lands.
        """

        # Show the environment
        self._env_show(path=self.path, what=['pos', 'obj'])

        # Take off
        self.controller.arm()
        self.controller.takeoff()

        # Execute actions
        self._execute()

        # Land
        self.controller.land()
        self.controller.disarm()

        # Keep environment open
        self.env.keep()


# - Naive algorithms used for tests; unusable in real life

class NaiveAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on optimal path finded and actions
    associated.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

    # Abstract interface

    def _is_keypoint(self, img):
        # Dummy implementation to respect abstract interface

        return False

    # Override how drone executes actions

    def _execute(self):
        # Group sequence of actions
        grouped = self.env.group_seq(self.sequence)

        # Execute each action
        for action, times in grouped:
            self.actions.get(action)(self._refs.get(action) * times)
            self.env.move(action, times)

            # Show updated environment
            self._env_show(path=self.path, what=['pos', 'obj'])


class VanishingAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on vanishing point detection method and
    environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Turn actions
        self.turn_actions = ['left', 'right']

    # Abstract interface

    def _is_keypoint(self, img):
        # Dummy implementation to respect abstract interface

        return False

    # Override how drone executes actions

    def _execute(self):
        # Execute action, adjusted with vanishing point
        for action in self.sequence:
            if action not in self.turn_actions:
                img = self.controller.picture()

                self.vanishing.run(img=img)

            self.actions.get(action)(self._refs.get(action))
            self.env.move(action)

            # Show updated environment
            self._env_show(path=self.path, what=['pos', 'obj'])


# - Real navigation algorithms, possibly usable in real life

class VisionAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses predictions of Deep Learning model to
    determine if drone has reached a key point or not.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Deep module
        self.deep = DeepModule(
            n_channels=2,
            model_id='densenet161',
            weights_pth='densenet161.pth',
            verbose=True
        )

        # Threshold of confidence
        self.threshold = 0.85

    # Abstract interface

    def _is_keypoint(self, img):
        pred = self.deep.run(img=img)

        return pred[1] > self.threshold


class MarkerAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses marker detection and decoding to determine
    if drone has reached a key point or not.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Marker module
        self.marker = MarkerModule(decoder_id='aruco_opencv', verbose=True)

        # Threshold on marker size
        self.threshold = 80

    # Abstract interface

    def _is_keypoint(self, img):
        decoded, pts = self.marker.run(img=img)
        size = 0 if pts is None else math.dist(pts[0], pts[2])

        return decoded == '1' and size > self.threshold


class DepthAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses depth map associated to an image to
    determine if drone has reached a key point or not.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Depth module
        self.depth = DepthModule(verbose=True)

        # Threshold on distance
        self.threshold = 1800

    # Abstract interface

    def _is_keypoint(self, img):
        distance = self.depth.run(img=img)

        return distance < self.threshold
