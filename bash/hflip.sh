#!/bin/bash

# Implementation of a tool used to horizontally flip images
# (and save them as new images in new folders).
#
# Usage: bash hflip.sh <dir_name>

#############
# Variables #
#############

if [ $# -lt 1 ]; then
    echo "Usage: bash hflip.sh <dir_name>"

    exit
fi

# Name of the directory to apply flip
DIR=$1


########
# Main #
########

count=0

for d in ${DIR}/*; do
    mkdir -p ${d}-flipped

    for f in ${d}/*.png; do
        base=$(basename -- "$f")
        base="${base%.*}"

        convert -flop $f ${d}-flipped/flipped_${base}_$count.png

        count=$((count + 1))
    done
done
