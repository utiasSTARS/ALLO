#!/bin/bash

GPU=$1
SING_IMG=$PWD/misc/blender.sif

VOLUMES="--bind=${PWD}/python:/home/src
         --bind=${PWD}/misc:/home/misc
         --bind=${PWD}/cad_models:/home/cad_models
         --bind=${PWD}/renders:/home/renders"

EXP_NUM=39
SEED=42

#? Start container
singularity instance start --writable-tmpfs --nv -e \
--no-home \
$VOLUMES \
$SING_IMG \
blender_exp${EXP_NUM}_${GPU}

# BASE_CMD="blender /home/misc/ephemeris_model_v3.blend --background --python render.py -- \
# --exp_num $EXP_NUM \
# --mode illumination depth scale \
# --anomaly \
# --seed $SEED"
BASE_CMD="blender /home/misc/ephemeris_model_v3_fb.blend --background --python render_binary.py -- \
--exp_num $EXP_NUM \
--mode illumination depth scale color \
--anomaly \
--seed $SEED"

CONTAINER_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU singularity exec --pwd /home/src \
instance://blender_exp${EXP_NUM}_$GPU \
$BASE_CMD"

eval $CONTAINER_CMD > ${PWD}/logs/gpu_$GPU.log 2>&1 &
# eval $CONTAINER_CMD