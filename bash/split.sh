#!/bin/bash

# Implementation of a tool used to split a small video into PNG images.
#
# Usage: bash split.sh <video> <output_dir>

#############
# Variables #
#############

if [ $# -lt 2 ]; then
    echo "Usage: bash split.sh <video> <output_dir>"

    exit
fi

# Path to video to split
VID=$1

# Path to output folder (where image will be saved)
OUT=$2


########
# Main #
########

# Create output folder
mkdir -p $OUT

# Split the video
ffmpeg -i $VID ${OUT}img_%06d.png
