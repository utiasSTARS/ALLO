#! /bin/bash

sudo SINGULARITY_NOHTTPS=1 singularity build -F misc/blender.sif docker-daemon://blender:latest