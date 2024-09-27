#! /bin/bash

VOLUMES="--volume=${PWD}/cad_models:/home/cad_models
        --volume=${PWD}/python:/home/src
        --volume=${PWD}/renders:/home/renders
        --volume=${PWD}/misc:/home/misc"

GPU='"device=0"'

docker run \
-it \
-p 6006:6006 \
--privileged \
-e DISPLAY=unix$DISPLAY \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
--gpus $GPU \
--shm-size 32G \
$VOLUMES \
--name=blender \
blender

# Add the following to docker run for gui:
# -e DISPLAY=unix$DISPLAY \
# -v /tmp/.X11-unix/:/tmp/.X11-unix/ \

# Run the following before docker run to allow x server connection
# xhost +local:root