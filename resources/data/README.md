# Data

All data used for this project are available via the following link : [data](https://drive.google.com/drive/folders/1fY7fqUh7_plg5A-T38kQFF37-xQE5Ubu?usp=sharing).

A data set has been created for each simulated environment.

A data set is organized as follows:

* folders `<data>/` that contain images related to "`data`" (for example, `vanishing-point/` contains images with annotated vanishing point);
* a folder `json/` that contains all annotations (for example, `json/vanishing-point.json` contains the annotations of all vanishing points).

> JSON files contain **relative links** to images. Make sure to adapt these links if you want to train Deep Learning models on a GPU cluster, for example!
