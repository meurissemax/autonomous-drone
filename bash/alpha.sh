#!/bin/bash

# Implementation of a tool used to remove alpha channel of PNG images.
# The image are modified in place.
#
# Usage: bash alpha.sh <dir_name>

#############
# Variables #
#############

if [ $# -lt 1 ]; then
    echo "Usage: bash alpha.sh <dir_name>"

    exit
fi

# Name of the directory to process
DIR=$1


########
# Main #
########

# Find all PNG images and remove alpha channel
find $DIR -name "*.png" -exec convert "{}" -alpha off "{}" \;
