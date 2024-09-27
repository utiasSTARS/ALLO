#!/bin/bash

sudo bash -c "echo '' > $(docker inspect --format="{{.LogPath}}" blender)"
docker attach --detach-keys="ctrl-a" blender