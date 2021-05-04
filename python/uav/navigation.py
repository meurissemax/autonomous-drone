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
from itertools import product
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

        # Define middle cell and length of grid side
        self.middle = math.floor(grid_size / 2)
        self.side = math.sqrt(grid_size)

        # Define left and right cells
        bounds = [-self.side, 0, self.side]

        self.left = [sum(x) for x in zip([self.middle + 1] * 3, bounds)]
        self.right = [sum(x) for x in zip([self.middle - 1] * 3, bounds)]

    # Abstract interface

    def _alignment(self, img):
        # Get cell that contains vanishing point
        pred = self.deep.run(img)
        cell = np.argmax(pred)

        # Get alignment
        alignment = 'center'

        if cell in self.left:
            alignment = 'left'
        elif cell in self.right:
            alignment = 'right'

        self.log(f'Alignment: {alignment}')

        return alignment


class DepthModule(NavModule):
    """
    Module that infers depth map associated to an image and compute mean depth
    of different areas in the image.
    """

    def __init__(self, n: int = 3, verbose=False):
        super().__init__(verbose)

        # Deep module
        self.deep = DeepModule(
            n_channels=2,  # dummy
            model_id='midas',
            weights_pth='',  # dummy
            verbose=False
        )

        # Grid size
        self.n = n

    # Abstract interface

    def run(self, img: Image) -> float:
        # Get depth map associated to image
        depth = self.deep.run(img=img)

        # Dimensions
        h, w = depth.shape

        # Split the image in cells and compute mean depth of each
        cell = (w // self.n, h // self.n)
        means = []

        for (j, i) in product(range(self.n), range(self.n)):
            left = i * cell[0]
            right = (i + 1) * cell[0]
            bottom = j * cell[1]
            top = (j + 1) * cell[1]

            cropped = depth[bottom:top, left:right]

            means.append(np.mean(cropped))

        self.log(f'Mean depths: {means}')

        return means


# - Staircase

class StaircaseNaiveModule(NavModule):
    """
    Module that moves the drone in a staircase (up or down).
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        self.controller = controller

    # Abstract interface

    def run(self, img: Image, direction: str):
        self.controller.move(direction, 30)
        self.controller.move('forward', 30)


class StaircaseDepthModule(NavModule):
    """
    Module that moves the drone in a staircase (up or down) based on relative
    depth map estimation.
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        self.controller = controller

        # Depth module
        self.depth = DepthModule(n=3, verbose=False)

        # Threshold
        self.threshold = 200

    # Abstract interface

    def run(self, img: Image, direction: str):
        while True:
            # Get mean depth estimations
            means = self.depth.run(img=img)

            # Compute depth of each zone
            means = np.array(means)

            top = np.mean(means[0:3])
            center = np.mean(means[3:6])
            bottom = np.mean(means[6:9])

            # Move the drone
            self.controller.move(direction, 30)

            if top + self.threshold > center >= bottom:
                self.controller.move(direction, 30)
            else:
                break

            # Take a new picture
            img = self.controller.picture()


class StaircaseMarkerModule(NavModule):
    """
    Module that moves the drone in a staircase (up or down) based on marker
    detection.
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        self.controller = controller

        # Marker module
        self.marker = MarkerModule(decoder_id='aruco_opencv', verbose=True)

        # (Known) width of the marker
        self.width = 10

        # Focal length
        self.f = 1540

    # Abstract interface

    def run(self, img: Image, direction: str):
        while True:
            # Decode marker, if any
            decoded, pts = self.marker.run(img=img)

            # If no marker is detected, stop
            if decoded is None:
                break

            # Perceived width
            p = pts[2][0] - pts[3][0]

            # Distance to marker
            d = (self.width * self.f) / p

            # Position of the middle points
            h, _, _ = img.shape

            mid_image = h // 2
            mid_marker = np.mean([pts[0][1], pts[2][1]])

            # Move the drone
            threshold = h // 10
            limits = [mid_image - threshold, mid_image + threshold]

            if mid_marker > limits[0] and mid_marker < limits[1]:
                self.controller.move('forward', max(0, d - 30))

            self.controller.move(direction, 30)

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

        self.staircase = StaircaseNaiveModule(
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

            # Check if drone has reach a key point
            #
            # To add the '+ Env' verification:
            # [...] and self.env.nearest_keypoint(self._keypoints) < 2:
            if self._is_keypoint(img):
                action, pos = self._keypoints[self._keypoints_idx]
                self._keypoints_idx += 1

                self._env_update(pos)
            else:
                # Align drone using vanishing point
                self.vanishing.run(img=img)

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
        self.threshold = 0.5

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

        # Focal length of the camera (to adapt according to the camera!)
        self.f = 1540

        # (Known) width of the marker used (in cm)
        self.width = 10

        # Threshold on distance (in cm)
        self.threshold = 100

    # Abstract interface

    def _is_keypoint(self, img):
        decoded, pts = self.marker.run(img=img)

        if decoded == '1':
            # Perceived width
            p = pts[2][0] - pts[3][0]

            # Distance to marker
            d = (self.width * self.f) / p

            return d <= self.threshold

        return False


class DepthAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses depth map associated to an image to
    determine if drone has reached a key point or not.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Grid size
        n = 3

        # Depth module
        self.depth = DepthModule(n=n, verbose=True)

        # Middle cell
        self.middle = math.floor((n ** 2) / 2)

        # Threshold
        self.threshold = 500

    # Abstract interface

    def _is_keypoint(self, img):
        means = self.depth.run(img=img)
        keypoint = False

        for idx, value in enumerate(means):
            if idx == self.middle:
                pass
            else:
                if means[self.middle] - value < self.threshold:
                    keypoint = True

        return keypoint


class VisionDepthAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses vision of the drone and depth estimation to
    determine if drone has reached a key point or not.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Vision algorithm
        self.vision = VisionAlgorithm(env, controller, show)

        # Depth algorithm
        self.depth = DepthAlgorithm(env, controller, show)

    # Abstract interface

    def _is_keypoint(self, img):
        return self.vision._is_keypoint(img) and self.depth._is_keypoint(img)


class VisionMarkerAlgorithm(NavAlgorithm):
    """
    Navigation algorithm that uses vision of the drone and marker detection to
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
        self.threshold = 0.8

        # Marker algorithm
        self.marker = MarkerAlgorithm(env, controller, show)

    # Abstract interface

    def _is_keypoint(self, img):
        pred = self.deep.run(img=img)

        if pred[1] >= self.threshold:
            return True
        elif pred[1] > 0.5 and pred[1] < self.threshold:
            return self.marker._is_keypoint(img)

        return False
