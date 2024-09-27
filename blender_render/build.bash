#!/bin/bash

DIR=$(dirname "$0")
DIR=${DIR%/}

docker build --no-cache \
    -t blender \
    -f $DIR/docker/Dockerfile \
    $DIR/docker