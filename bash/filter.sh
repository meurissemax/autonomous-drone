#!/bin/bash

# Implementation of a tool used to delete images. The script keep one image
# over 'keep' and delete the others.
#
# Usage: bash filter.sh <dir_name> <keep>

#############
# Variables #
#############

if [ $# -lt 2 ]; then
    echo "Usage: bash filter.sh <dir_name> <keep>"

    exit
fi

# Name of the directory to filter
DIR=$1

# Keep one image over 'keep'
keep=$2


########
# Main #
########

for d in ${DIR}/*; do
    count=0

    for f in ${d}/*.png; do
        mod=$((count % keep))
        count=$((count + 1))

        if [ $mod != 0 ]; then
            rm $f
        fi
    done
done
