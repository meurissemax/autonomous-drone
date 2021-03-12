"""
Implementation of navigation processes of the drone.
"""

###########
# Imports #
###########

import torch
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from functools import partial

from analysis.vanishing_point import VPClassic
from .controllers import Controller
from .environment import Environment
from learning.models import DenseNet161


###########
# Classes #
###########

class Navigation(ABC):
    """
    Abstract class to define a navigation algorithm.

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

    @abstractmethod
    def navigate(self):
        pass


class Naive(Navigation):
    """
    Naive navigation algorithm based solely on A* path finding.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # List of actions
        self.actions = {
            'forward': lambda t: self.controller.move('forward', 100 * t),
            'left': lambda t: self.controller.rotate('ccw', 90 * t),
            'right': lambda t: self.controller.rotate('cw', 90 * t)
        }

    def navigate(self):
        # Determine shortest path to objective
        path = self.env.path()

        # Get list of actions to follow the path
        actions = self.env.path_to_seq(path)
        actions = self.env.group_seq(actions)

        # Show the environment
        if self.show:
            self.env.render(path=path, what=['pos', 'obj'])

        # Initialize the drone
        self.controller.arm()
        self.controller.takeoff()

        # Execute each action
        for action, times in actions:
            self.actions.get(action)(times)
            self.env.move(action, times)

            # Show updated environment
            if self.show:
                self.env.render(path=path, what=['pos', 'obj'])

        # Stop the drone
        self.controller.land()
        self.controller.disarm()

        # Keep environment open
        if self.show:
            self.env.keep()


class Vision(Navigation):
    """
    Navigation algorithm based on the drone vision.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.device = device

        print(f'Device: {device}')

        # Choices
        self.choices = ['forward', 'turn']

        # Model
        weights_pth = 'weights.pth'

        model = DenseNet161(3, len(self.choices))
        model = model.to(self.device)
        model.load_state_dict(torch.load(weights_pth, map_location=device))
        model.eval()

        self.model = model

        # Actions
        self.actions = {
            'forward': partial(self.controller.move, 'forward', 100),
            'left': partial(self.controller.rotate, 'ccw', 90),
            'right': partial(self.controller.rotate, 'cw', 90)
        }

    def navigate(self):
        # Show the environment
        if self.show:
            self.env.render(what=['pos', 'obj'])

        # Initialize the drone
        self.controller.arm()
        self.controller.takeoff()

        # Initialize action
        action = None

        # Execute action based on drone vision
        while not self.env.has_reached_obj():
            img = self.controller.picture()
            img = transforms.ToTensor()(img)

            # Infer action
            with torch.no_grad():
                img = img.unsqueeze(0)
                img = img.to(self.device)

                output = self.model(img)
                output = output.squeeze(0)

                action = self.choices[torch.argmax(output)]

            self.actions.get(action)()
            self.env.move(action)

            # Show updated environment
            if self.show:
                self.env.render(what=['pos', 'obj'])

        # Stop the drone
        self.controller.land()
        self.controller.disarm()

        # Keep environment open
        if self.show:
            self.env.keep()


class Vanishing(Navigation):
    """
    Navigation algorithm based on the vanishing point detection.
    """

    def __init__(self, env, controller, show=False):
        super().__init__(env, controller, show)

        # List of actions
        self.actions = {
            'forward': partial(self.controller.move, 'forward', 100),
            'left': partial(self.controller.rotate, 'ccw', 90),
            'right': partial(self.controller.rotate, 'cw', 90)
        }

        # List of turn actions
        self.turn_actions = ['left', 'right']

        # List of adjustments
        self.adjustments = {
            'left': partial(self.controller.rotate, 'cw', 5),
            'right': partial(self.controller.rotate, 'ccw', 5)
        }

    def _alignment(self, lower: tuple, upper: tuple) -> str:
        # Take a picture
        img = self.controller.picture()

        # Get vanishing point
        x, _ = VPClassic.detect(img)

        # Check alignment
        if upper[0] < x < upper[1]:
            return 'left'
        elif lower[0] < x < lower[1]:
            return 'right'

        return 'center'

    def _adjust(self, mean: float, std: float):
        # Get lower and upper intervals
        lower = (mean - 2 * std, mean - std)
        upper = (mean + std, mean + 2 * std)

        # Get drone alignment
        alignment = self._alignment(lower, upper)

        # Adjust the drone
        while alignment != 'center':
            self.adjustments.get(alignment)()

            alignment = self._alignment(lower, upper)

    def navigate(self):
        # Determine shortest path to objective
        path = self.env.path()

        # Get list of actions to follow the path
        actions = self.env.path_to_seq(path)

        # Show the environment
        if self.show:
            self.env.render(path=path, what=['pos', 'obj'])

        # Initialize the drone
        self.controller.arm()
        self.controller.takeoff()

        # Get image dimensions
        img = self.controller.picture()
        h, w, _ = img.shape

        # Define bound parameters
        mean, std = w / 2, w / 20

        # Execute each action
        for action in actions:
            if action not in self.turn_actions:
                self._adjust(mean, std)

            self.actions.get(action)()
            self.env.move(action)

            # Show updated environment
            if self.show:
                self.env.render(path=path, what=['pos', 'obj'])

        # Stop the drone
        self.controller.land()
        self.controller.disarm()

        # Keep environment open
        if self.show:
            self.env.keep()
