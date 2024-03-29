# Autonomous navigation of a drone in indoor environments

This work is a research and development project on autonomous navigation algorithms for a small programmable drone in indoor environments free of dynamic obstacles. The hypothesis that the drone has access to a simple representation of its environment, in order to plan paths and analyze them, was posed.

## Context

Master's thesis carried out to obtain the degree of Master of Science in Computer Science Engineering ([University of Liège](https://uliege.be/), [Faculty of Applied Science](https://facsa.uliege.be/), academic year 2020-2021).

## Main objective

The main objective of this work was to explore and test modern techniques allowing the autonomous navigation of a drone in an indoor environment.

The main techniques explored are:

* the calculation of vanishing point to adjust the drone;
* the use of neural networks for various tasks (associating an action to an image, computing vanishing point, depth estimation);
* the use of markers (ArUco, QR code) to guide the drone.

Tests were first performed on a simulator ([Unreal Engine 4](https://www.unrealengine.com/) with the [AirSim](https://microsoft.github.io/AirSim/) plugin) and then in the corridors of a building with a [DJI Tello EDU](https://www.ryzerobotics.com/tello-edu).

## Resources

### Data

The configuration of the AirSim plugin as well as the simulated environments created on Unreal Engine 4 are described and available [here](resources/simulator/README.md).

The data sets constituted on each of the simulated environments are described and available [here](resources/data/README.md).

Examples of environment representations, used by the algorithms, are available [here](resources/environments/).

### Code

The [`bash/`](bash/) folder contains miscellaneous scripts used mainly to process images.

The main implementation of the algorithms is located in the [`python/`](python/) folder. More precisely,

* [`airsim/`](python/airsim/) is the AirSim Python package;
* [`analysis/`](python/analysis/) is a module containing implementation of marker detection and decoding algorithms and vanishing point detection algorithms;
* [`learning/`](python/learning/) is a module containing implementation of all elements related to Deep Learning (data sets, models, training and testing procedures);
* [`misc/`](python/misc/) contains miscellaneous scripts, mainly tests used to display and evaluate intermediate results;
* [`plots/`](python/plots/) is a module containing settings to create LaTeX plots with `matplotlib` package;
* [`uav/`](python/uav/) is the main module that contains implementation of the controllers, environnement representation and all navigation modules and algorithms.

Files [`learn.py`](python/learn.py) and [`navigate.py`](python/navigate.py) are main files used to, respectively, train and evaluate a neural network and navigate the drone.

## Results

All results obtained are explicited in my [thesis report](latex/main.pdf).

Important results are illustrated by videos available on [this YouTube playlist](https://youtube.com/playlist?list=PLJEcTQrQgiVdacuc2HymqLV9RqjaRMNYt).

## Try it yourself

First, make sure to create and activate the Anaconda environment using

```bash
conda env create -f environment.yml
conda activate autonomous-drone
```

Then, you can simply run the [`navigate.py`](python/navigate.py) script using, for example,

```bash
python navigate.py --environment my_env.txt --controller airsim --algorithm vision --show
```

Help with arguments can be obtained via

```bash
python navigate.py --help
```

### Custom environment

If you want to work with your (simulated or real) environment, make sure to create a representation as described in the `Environment` class of the [`environment.py`](python/uav/environment.py) file. Examples can be found [here](resources/environments/).

### Deep Learning

If you want to use navigation algorithms that use neural network(s), make sure to train model(s) using the [`learn.py`](python/learn.py) script.

Help with arguments can be obtained via

```bash
python learn.py --help
```

Exemples of ready-to-use data sets are available [here](resources/data/README.md).

## References

All references used are listed in my [thesis report](latex/main.pdf).
