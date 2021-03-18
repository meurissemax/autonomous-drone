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

from analysis.qr_code import QRZBar
from analysis.vanishing_point import VPClassic
from .controllers import Controller
from .environment import Environment
from learning.models import DenseNet, SmallConvNet


##########
# Typing #
##########

Information = Union[List, str, Tuple]
Image = np.array


###########
# Classes #
###########

# Abstract classes

class NavModule(ABC):
    """
    Abstract class used to define a navigation module.

    A navigation module is a component of a navigation algorithm. It performs
    several actions on drone input(s) and returns some information. It never
    performs directly an action on the drone!

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
    def run(self, **kwargs) -> Information:
        """
        Take as inputs some data of the drone (image, IMU data, etc) and return
        some information (action, position, etc).
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
    Module that determines probabilities of next action of the drone based on
    Deep Learning model predictions.
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

        # Processing
        self.process = transforms.Compose([
            lambda x: cv2.resize(x, (320, 180)),
            transforms.ToTensor()
        ])

    # Abstract interface

    def run(self, img: Image):
        img = self.process(img)

        with torch.no_grad():
            img = img.unsqueeze(0)
            img = img.to(self.device)

            pred = self.model(img)
            pred = torch.flatten(pred)

            self.log(f'Prediction: {pred}')

            return pred.tolist()


class VanishingModule(NavModule):
    """
    Module that determines orientation of the drone based on vanishing point
    detection method.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)

        # Vanishing point detector
        self.method = VPClassic()

        # Bounds
        self.lower, self.upper = None, None

    # Abstract interface

    def run(self, img: Image):
        img = cv2.resize(img, (320, 180))

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


class QRCodeModule(NavModule):
    """
    Module that reads the content of a QR code, if any, and returns it along
    with coordinates of its corner points.
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)

        # QR code decoder
        self.decoder = QRZBar()

    # Abstract interface

    def run(self, img: Image):
        decoded, pts = self.decoder.decode(img)

        self.log(f'Decoded: {decoded}')

        return decoded, pts


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


class VisionAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on prediction by Deep Learning model on
    drone's vision (picture taken) and environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create vision module
        self.module = VisionModule(
            n_outputs=2,
            model_id='densenet121',
            weights_pth='densenet121.pth',
            verbose=True
        )

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
            pred = self.module.run(img=img)

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


class VanishingAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on vanishing point detection method and
    environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create vanishing point module
        self.module = VanishingModule(verbose=True)

        # List turn actions
        self.turn_actions = ['left', 'right']

        # Define adjustment actions
        self.adjustments = {
            'left': partial(self.controller.rotate, 'cw', 5),
            'right': partial(self.controller.rotate, 'ccw', 5)
        }

    # Abstract interface

    def _execute(self):
        # Execute action, adjusted with vanishing point
        for action in self.sequence:
            if action not in self.turn_actions:
                while True:
                    img = self.controller.picture()
                    alignment = self.module.run(img=img)

                    if alignment == 'center':
                        break

                    self.adjustments.get(alignment)()

            self.actions.get(action)(self.steps.get(action))
            self.env.move(action)

            # Show updated environment
            self._show(path=self.path, what=['pos', 'obj'])


class QRCodeAlgorithm(NavAlgorithm):
    """
    Navigation algorithm based solely on QR code detection and decoding and
    environment information.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Create QR code module
        self.module = QRCodeModule(verbose=True)

        # Threshold on QR code size
        self.threshold = 120

    # Abstract interface

    def _execute(self):
        # Get key points
        keypoints = self.env.extract_keypoints(self.path, self.sequence)
        idx = 0

        # Execute actions, guided by QR codes
        while not self.env.has_reached_obj():
            img = self.controller.picture()
            decoded, pts = self.module.run(img=img)

            # Compute QR code size
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
