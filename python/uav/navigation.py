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

from analysis.markers import ArucoOpenCV, QRZBar
from analysis.vanishing_point import VPClassic
from .controllers import Controller
from .environment import Environment
from learning.models import DenseNet, SmallConvNet


##########
# Typing #
##########

Image = np.array


###########
# Classes #
###########

# Abstract classes

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
        self.show = show

        # Path and associated sequence of actions
        self.path = env.path()
        self.sequence = env.path_to_seq(self.path)

        # Possible actions of the drone
        self.actions = {
            'forward': lambda d: self.controller.move('forward', d),
            'left': lambda a: self.controller.rotate('ccw', a),
            'right': lambda a: self.controller.rotate('cw', a)
        }

        # Define steps for each action
        self.steps = {
            'forward': 100,
            'left': 90,
            'right': 90
        }

    def _show(self, **kwargs):
        """
        Show the environment representation if flag is enabled.
        """

        if self.show:
            self.env.render(**kwargs)

    @abstractmethod
    def _execute(self):
        """
        Define how the drone executes actions during the navigation process.

        Must be defined by each navigation algorithm.
        """

        pass

    def navigate(self):
        """
        Move automatically the drone from its initial position to an objective.

        This navigation process follow a general template: the drone takes off,
        executes actions and lands.
        """

        # Show the environment
        self._show(path=self.path, what=['pos', 'obj'])

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


# Modules

class VisionModule(NavModule):
    """
    Module that determines probabilities of an event based on Deep Learning
    model predictions.
    """

    def __init__(
        self,
        n_outputs: int,
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
            'small': SmallConvNet
        }

        model = models.get(model_id)(3, n_outputs)
        model = model.to(device)
        model.load_state_dict(torch.load(weights_pth, map_location=device))
        model.eval()

        self.model = model

        # GPU warm-up
        for _ in range(10):
            _ = self.model(torch.randn(3, 180, 320).to(self.device))

        # Processing
        self.process = transforms.Compose([
            lambda x: cv2.resize(x, (320, 180)),
            transforms.ToTensor()
        ])

    # Abstract interface

    def run(self, img: Image) -> List:
        img = self.process(img)

        with torch.no_grad():
            img = img.unsqueeze(0)
            img = img.to(self.device)

            pred = self.model(img).cpu()
            pred = torch.flatten(pred)

            self.log(f'Prediction: {pred}')

            return pred.tolist()


class VanishingModule(NavModule):
    """
    Abstract class used to define navigation module that works with vanishing
    point.
    """

    def __init__(self, controller: Controller, verbose=False):
        super().__init__(verbose)

        # Controller
        self.controller = controller

        # Define adjustment actions
        self.adjustments = {
            'left': partial(self.controller.rotate, 'cw', 5),
            'right': partial(self.controller.rotate, 'ccw', 5)
        }

    @abstractmethod
    def _alignment(self, img: Image) -> str:
        """
        Determine the orientation of the drone (left, right or center) based on
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


class VanishingCVModule(VanishingModule):
    """
    Module that adjusts the drone by estimating its alignment based on a
    vanishing point detection using pure computer vision method.
    """

    def __init__(self, controller, verbose=False):
        super().__init__(controller, verbose)

        # Vanishing point detector
        self.method = VPClassic()

        # Bounds
        self.lower, self.upper = None, None

    # Abstract interface

    def _alignment(self, img):
        # Define bounds, if not defined
        if self.lower is None:
            h, w, _ = img.shape
            mean, std = w / 2, w / 20

            self.lower = (mean - 2 * std, mean - std)
            self.upper = (mean + std, mean + 2 * std)

        # Detect vanishing point
        x, _ = self.method.detect(img)

        # Get orientation
        orientation = 'center'

        if self.upper[0] < x < self.upper[1]:
            orientation = 'right'
        elif self.lower[0] < x < self.lower[1]:
            orientation = 'left'

        self.log(f'Orientation: {orientation}')

        return orientation


class VanishingDLModule(VanishingModule):
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

        # Vision module
        self.vision = VisionModule(grid_size, model_id, weights_pth, False)

        # Define middle cell
        self.middle = math.floor(grid_size / 2)

    # Abstract interface

    def _alignment(self, img):
        # Get cell that contains vanishing point
        pred = self.vision.run(img)
        cell = pred.index(max(pred))

        # Get orientation
        orientation = 'center'

        if cell == self.middle + 1:
            orientation = 'right'
        elif cell == self.middle - 1:
            orientation = 'left'

        self.log(f'Orientation: {orientation}')

        return orientation


class MarkerModule(NavModule):
    """
    Module that reads the content of a marker, if any, and returns it along
    with coordinates of its corner points.
    """

    def __init__(self, marker: str, verbose=False):
        super().__init__(verbose)

        # Marker decoder
        decoders = {
            'aruco': ArucoOpenCV,
            'qr': QRZBar
        }

        self.decoder = decoders.get(marker)()

    # Abstract interface

    def run(self, img: Image) -> Tuple[str, List]:
        decoded, pts = self.decoder.decode(img)

        self.log(f'Decoded: {decoded}')

        return decoded, pts


class DepthModule(NavModule):
    """
    Module that infers depth map associated to an image and compute distance to
    element in front of the drone.

    For the moment, in order to perform tests, the module works DIRECTLY with a
    ground truth depth map obtained via AirSim.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)

    # Abstract interface

    def run(self, img: Image) -> float:
        # Dimensions
        h, w = img.shape

        # Crop the image to keep center box
        mx, my = w // 2, h // 2
        dx, dy = w // 10, h // 10

        cropped = img[(mx - dx):(mx + dx), (my - dy):(my + dy)]

        # Compute distance
        distance = np.mean(cropped)

        self.log(f'Distance: {distance}')

        return distance


class StaircaseModule(NavModule):
    """
    Module that moves the drone in a staircase (up or down) based on an image.
    """

    def __init__(self, controller: Controller, direction: str, verbose=False):
        super().__init__(verbose)

        # Controller
        self.controller = controller

        # Direction
        self.direction = direction

        # Depth module
        self.depth = DepthModule(False)

        # Threshold on distance
        self.threshold = 1

        # Limit to consider the staircase
        self.limit = 2

        # Define staircase actions
        self.staircase = {
            'up': partial(self.controller.move, 'up', 30),
            'down': partial(self.controller.move, 'down', 30),
            'forward': partial(self.controller.move, 'forward', 30)
        }

    # Abstract interface

    def run(self, img: Image):
        while True:
            # Get distance to staircase
            distance = self.depth.run(img=img)

            self.log(f'Distance: {distance}')

            # Check limit
            if distance > self.limit:
                break

            # Move the drone, according to distance
            if distance < self.threshold:
                self.staircase.get(self.direction)()
            else:
                self.staircase.get('forward')()

            # Take a new picture
            img = self.controller.picture()


# Navigation algorithms

class NaiveAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on optimal path finded and actions
    associated.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

    # Abstract interface

    def _execute(self):
        # Group sequence of actions
        grouped = self.env.group_seq(self.sequence)

        # Execute each action
        for action, times in grouped:
            self.actions.get(action)(self.steps.get(action) * times)
            self.env.move(action, times)

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])


class VanishingAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on vanishing point detection method and
    environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create vanishing point module
        self.vanishing = VanishingCVModule(controller=controller, verbose=True)

        # List turn actions
        self.turn_actions = ['left', 'right']

    # Abstract interface

    def _execute(self):
        # Execute action, adjusted with vanishing point
        for action in self.sequence:
            if action not in self.turn_actions:
                img = self.controller.picture()

                self.vanishing.run(img=img)

            self.actions.get(action)(self.steps.get(action))
            self.env.move(action)

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])


class VisionAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based on predictions by Deep Learning model on drone's
    vision (picture taken), vanishing point detection and environment
    information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create vision module
        self.vision = VisionModule(
            n_outputs=2,
            model_id='densenet121',
            weights_pth='densenet121.pth',
            verbose=True
        )

        # Create vanishing point module
        self.vanishing = VanishingCVModule(controller=controller, verbose=True)

        # Threshold of confidence
        self.threshold = 0.85

    # Abstract interface

    def _execute(self):
        # Get key points
        keypoints = self.env.extract_keypoints(self.path, self.sequence)
        idx = 0

        # Execute actions based on predictions
        while not self.env.has_reached_obj():
            img = self.controller.picture()

            # Align drone with vanishing point
            self.vanishing.run(img=img)

            # Get predicition of Deep Learning model
            pred = self.vision.run(img=img)

            # Check if it is a turn or not
            if pred[1] > self.threshold:
                action, pos = keypoints[idx]
                idx += 1

                self.env.update(pos, action)
            else:
                action = 'forward'

                self.env.move(action)

            self.actions.get(action)(self.steps.get(action))

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])


class MarkerAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based on marker detection and decoding, vanishing
    point detection and environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create marker module
        self.qr = MarkerModule(marker='qr', verbose=True)

        # Create vanishing point module
        self.vanishing = VanishingCVModule(controller=controller, verbose=True)

        # Threshold on marker size
        self.threshold = 140

    # Abstract interface

    def _execute(self):
        # Get key points
        keypoints = self.env.extract_keypoints(self.path, self.sequence)
        idx = 0

        # Execute actions, guided by markers
        while not self.env.has_reached_obj():
            img = self.controller.picture()

            # Align drone with vanishing point
            self.vanishing.run(img=img)

            # Decode marker, if any
            decoded, pts = self.qr.run(img=img)

            # Compute marker size
            size = 0 if pts is None else math.dist(pts[0], pts[2])

            # Check if it is a turn or not
            if decoded == 'turn' and size > self.threshold:
                action, pos = keypoints[idx]
                idx += 1

                self.env.update(pos, action)
            else:
                action = 'forward'

                self.env.move(action)

            self.actions.get(action)(self.steps.get(action))

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])


class DepthAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based on depth map (and so, approximate distances to
    elements) associated to an image.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create depth module
        self.depth = DepthModule(verbose=True)

        # Create vanishing point module
        self.vanishing = VanishingCVModule(controller=controller, verbose=True)

        # Threshold on distance
        self.threshold = 2

    # Abstract interface

    def _execute(self):
        # Get key points
        keypoints = self.env.extract_keypoints(self.path, self.sequence)
        idx = 0

        # Execute actions based on predictions
        while not self.env.has_reached_obj():
            img = self.controller.picture()

            # Align drone with vanishing point
            # self.vanishing.run(img=img)

            # Get distance to the element in front of the drone
            distance = self.depth.run(img=img)

            # Check if it is a turn or not
            if distance < self.threshold:
                action, pos = keypoints[idx]
                idx += 1

                self.env.update(pos, action)
            else:
                action = 'forward'

                self.env.move(action)

            self.actions.get(action)(self.steps.get(action))

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])
