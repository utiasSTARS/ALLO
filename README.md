# ALLO
**ALLO: A Photorealistic Dataset and Data Generation Pipeline for Anomaly Detection During Robotic Proximity Operations in Lunar Orbit** 

Selina Leveugle, Chang Won Lee, Svetlana Stolpner, Chris Langley, Paul Grouchy, Steven Waslander, and Jonathan Kelly

## Introduction
This repository provides the data generation pipeline to generate and expand the ALLO dataset. 
The ALLO dataset and all supporting Blender files are available for download [here](https://trail-super-nas.synology.me:61536/fsdownload/Wy3jhzOeH/ALLO), and the Anomalib benchmark code will be released upon final acceptance.
The preprint is available at [https://arxiv.org/abs/2409.20435](https://arxiv.org/abs/2409.20435)


## Blender Render
### Installation
The Dockerfile, corresponding requirements, and .bash files are provided for installation using docker.
To install on Ubuntu >=20.04 follow the steps outlined below. It is important to the note that setting up the environment for Python in Blender can be challenging. Packages must be installed directly in Blender's Python directory for Blender to be able to import them. 

1. Install Blender from [here](https://www.blender.org/download/)

	a) Use Blender version > 2.9

	b) untar the downloaded zip folder and make note of the path to the Blender folder, the folder structure will look like this:
	```
	./blender-3.6.5-linux-x64
	├── blender
	├── 3.6
	...
	```
	where `blender` is the executable used to run Blender, and `3.6` is the blender version which contains Blender's Python environment

	c) Install Blender using apt to install dependencies
	```
	apt install blender
	```

2. install requirements 

	a) general requirements can be installed with:
	```
	pip install -r requirements.txt
	```
	b). install blender requirements inside the Blender Python environment by running:
```
/blender-3.6.5-linux-x64/3.6/python/bin/python3.10 -m pip install -r requirement_bpy.txt
```

3. Download Blender-Render code, including the ephemeris csv file
```
git clone https://github.com/utiasSTARS/ALLO.git
```

4. Download all required Blender files from [here](http://gofile.me/5fLQD/mL39QLs3n). Required files are the ephemeris_model.blend, the CAD models folder with models of the Moon, Earth, and Sun, and the folder with anomaly CAD models


### Usage
Images are created using the `render_binary.py` script which uses the following flags:
- `exp_num` is the experiment number, all images are saved to a folder named `exp_<exp_num>`
- `anomaly` is a boolean flag that determines if the rendered images are normal (False) or anomalous (True), default is False
- `mode` lists the various experiment modes that are used during the rendering: `illumination` renders multiple with different sun strengths, `depth` changes the anomaly depth, `scale` randomly sets the scale of the anomaly, and `color` sets the anomaly to a random colour 
- `seed` specifies the experiment seed
- `config` is the path to `render_config.yaml` where various scene parameters, including start and end days, cameras, mode parameters, and anomalies, are specified


**To create normal (anomaly-free) images**
1. In `render_config.yaml` file specify the start day, end day, and day interval, as well as sun strengths values if varying illumination. Set paths to ephemeris csv file, CAD model folders, anomalies model folder, and specify output path where `exp_<exp_num>` is saved
2. Run the following command from the python folder in Blender-Render while specifying where Blender was installed to run, where the ephemeris_model.blend file is saved, and the config file
```
../blender-3.6.5-linux-x64/blender ../ephemeris_model.blend --background --python render_binary.py -- \ -- 
	--config render_config.yaml \
	--exp_num 1  \
	--mode illumination
```

this will create anomaly-free images with different illumination values in a folder called `exp_1` with the following structure:
```
exp_1/
├── Camera1/
│   ├── fb_mask/
│   └── normal/
├── Camera2/
│   ├── fb_mask/
│   └── normal/
├── Camera3/
│   ├── fb_mask/
│   └── normal/
...
```


**To create anomalous images**
1. In `render_config.yaml` file specify the start day, end day, and day interval, as well as any varied parameters such as sun strength, colour, scale, or depth. Set paths to ephemeris csv file, cad model folders, anomalies model folder, and specify output path where `exp_<exp_num>` is saved
2. Run the following command from the python folder in Blender-Render while specifying where Blender was installed to run, where the ephemeris_model.blend file is saved, and the config file, as well as any modes of scene variation
```
../blender-3.6.5-linux-x64/blender ../ephemeris_model_v3_fb.blend --background --python render_binary.py -- \ -- 
	--config render_config.yaml \
	--exp_num 2 \
	--mode illumination depth scale color \
	--anomaly
```

this will create anomalous images with different illumination, depths, randomly scaled and coloured anomalies in a folder called `exp_2` with the following structure:
```
exp_2/
├── Camera1/
│   ├── anomaly/
│   ├── anomaly_mask/
│   ├── fb_mask/
├── Camera2/
│   ├── anomaly/
│   ├── anomaly_mask/
│   ├── fb_mask/
├── Camera3/
│   ├── anomaly/
│   ├── anomaly_mask/
│   ├── fb_mask/
...
```


**Processing the images**

The rendered images are then verified and reorganized such that they can be used in the Anomalib benchmark. This is done using the `preprocess.py` script.
In this script, anomalous images are verified to ensure that they are the correct shape and that anomalies meet the minimum pixel requirement. The foreground/background mask is combined with the anomaly map to output a 3-class segmentation mask. A log file lists any issues with the images or masks. For normal images the images are verified to ensure the correct shape and saved to correctly name folders. The script uses the following flags:
- `input` path to the `exp_<exp_num>` folder generated using the rendering script
- `output` path to output folder where verified and reorganized images will be saved
- `log_file` path to a `.log` file that logs the image processing (this file does not have to exist before running the script)
- `seed` the seed that was used to render the images in the input folder
- `anomaly` boolean for anomalous (True) or normal (False) images, default is False
- `min_pixel` is the required minimum pixel size for an anomaly 
To run the preprocessing script on normal images:
```
python3 preprocess.py 
	--input ../renders/exp_1/ \
	--output ../renders/train/ \
	--log_file ../renders/exp1log.log \
	--seed 42
```
This outputs a folder of normal images with the following structure:
```
train/
├── Camera1/
│   ├── images/
│   ├── masks/
├── Camera2/
│   ├── images/
│   ├── masks/
├── Camera3/
│   ├── images/
│   ├── masks/
...
```

To run the preprocessing script on anomalous images:
```
python3 preprocess.py 
	--input ../renders/exp_2/ \
	--output ../renders/test/ \
	--log_file ../renders/exp2log.log \
	--seed 42 \
	--anomaly \
	--min_pixel 2000
```
This outputs a folder of anomalous images with the structure:
```
test/
├── Camera1/
│   ├── images/
│   ├── masks/
├── Camera2/
│   ├── images/
│   ├── masks/
├── Camera3/
│   ├── images/
│   ├── masks/
...
```
Normal images can be added to the test output folder depending on the desired ratio of normal/anomalous images during testing. This pre-processing steps organizes image such that they can be used by the Anomalib benchmark evaluation code.



