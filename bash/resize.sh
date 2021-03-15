#!/bin/bash

# Implementation of a tool used to resize PNG images.
# The image are modified in place.
#
# Usage: bash resize.sh <dir_name> <width> <height>

#############
# Variables #
#############

if [ $# -lt 3 ]; then
    echo "Usage: bash resize.sh <dir_name> <width> <height>"

    exit
fi

# Name of the directory to process
DIR=$1

# New dimensions
WIDTH=$2
HEIGHT=$3


########
# Main #
########

# Find all PNG images and resize them
find $DIR -name "*.png" -exec convert "{}" -resize ${WIDTH}x${HEIGHT}\! "{}" \;
