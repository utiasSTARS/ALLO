import argparse
import os
import os.path as osp
from pathlib import Path
import shutil
import sys
import bpy
from collections import OrderedDict
import numpy as np
import cv2
import random
from tqdm import tqdm
import itertools as it
import yaml

#* add path to sys for importing modules
#* ------------------------------------------------
sys.path.append(os.path.dirname(bpy.data.filepath))
sys.path.append(os.getcwd())

from utils import (
    setup, load_models, setup_cams, get_cam_pos, 
    load_anomaly, set_anomaly_position, set_anomaly_scale, 
    set_anomaly_colour
)
#* ------------------------------------------------

# possible anomaly colours
COLOURS = {
    "red": (1,0,0,1),
    "green": (0,1,0,1),
    "blue": (0,0,1,1),
    "yellow": (1,1,0,1),
    "purple": (1,0,1,1),
    "cyan": (0,1,1,1),
    "white": (1,1,1,1),
}

def setup_sunlight(strength, station):
    '''Add blender sun light
    
    Args:
        - strenght: sun light strenght in blender units
        - station: the iss station model
    '''
    # add blender sun-type light to model
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', 
        location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].name = "sun light"
    sun_light = bpy.data.objects['sun light']
    # set sun light to track station so the light is pointing from the sun's position
    ttc = sun_light.constraints.new(type='TRACK_TO')
    ttc.target = station
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_X'
    # sun light parameters (strenght and angle)
    sun_light.data.energy = strength
    sun_light.data.angle = 0.010472

def load_ephemeris(ephemeris_path):
    '''get ephemeris data for positions of moon, earth, sun
    
    Args: 
        - ephemeris_path: path to csv with radius and position data of moon, earth, and sun
        
    Returns:
        - dictionary with moon, sun, and earth ephemeris data
    '''
    ephemeris_data = np.loadtxt(ephemeris_path, delimiter=',')
    # ephemeris data is 365x12 array for 1 year of data
    moon_pose = ephemeris_data[:, :4] # moon radius, x, y, z
    sun_pose = ephemeris_data[:, 4:8] # sun radius, x, y, z
    earth_pose = ephemeris_data[:, 8:12] # earth radius, x, y, z
    return {
        "moon": moon_pose,
        "sun": sun_pose,
        "earth": earth_pose
    }

def render_anomaly_single(output_dir, prefix, anomaly_path, fb_mask_path, anomaly_mask_path, day):
    '''render an anomalous image
    
    Args:
        - output_dir: path to camera folder where renders are saved
        - prefix: image file name with scene parameters
        - anomaly_path: name of anomalous folder where anomalous images are saved
        - fb_mask_path: name of foreground/background folder where mask is saved
        - anomaly_mask_path: name of anomaly mask folder where anomaly mask is saved
        - day: current iteration of ephemeris data
    '''
    bpy.ops.render.render(write_still=True)
    # Rename files (bug in file output node)
    for pth in [anomaly_path, fb_mask_path, anomaly_mask_path]:
        file_name = os.listdir(osp.join(output_dir, pth, str(day)))[0]
        old_file_name = osp.join(output_dir, pth, str(day), file_name)
        new_file_name = osp.join(output_dir, pth, f"{day}_{prefix}.png")
        shutil.move(old_file_name, new_file_name)
        shutil.rmtree(osp.join(output_dir, pth, str(day)))

def render_normal_single(output_dir, prefix, normal_path, fb_mask_path, day):
    '''render normal image
    
    Args:
        - output_dir: path to camera folder where renders are saved
        - prefix: image file name with scene parameters
        - normal_path: name of folder where normal images are saved
        - fb_mask_path: name of foreground/background folder where mask is saved
        - day: current iteration of ephemeris data
    '''
    bpy.ops.render.render(write_still=True)
    # Rename files (bug in file output node)
    for pth in [normal_path, fb_mask_path]:
        file_name = os.listdir(osp.join(output_dir, pth, str(day)))[0]
        old_file_name = osp.join(output_dir, pth, str(day), file_name)
        new_file_name = osp.join(output_dir, pth, f"{day}_{prefix}.png")
        shutil.move(old_file_name, new_file_name)
        shutil.rmtree(osp.join(output_dir, pth, str(day)))


def check_anomaly_pixels(output_dir, mask_path, day, min_pixels):
    """Check if anomaly is of sufficient size.
    
    Args:
        - output_dir: path to camera folder where renders are saved
        - mask_path: name of folder where anomaly mask are saved
        - day: current iteration of ephemeris data
        - min_pixels (int): minimum number of pixels for anomaly to be valid
    
    Returns:
        - bool: True if anomaly is valid, False otherwise
    """
    # Get nodes from tree
    node_tree = bpy.data.scenes['Scene'].node_tree
    render_layers_node = node_tree.nodes["Render Layers"]
    denoise_node = node_tree.nodes["Denoise"]
    
    # Remove node links temporarily
    node_tree.links.remove(render_layers_node.outputs["Noisy Image"].links[0])
    node_tree.links.remove(render_layers_node.outputs["Denoising Normal"].links[0])
    node_tree.links.remove(render_layers_node.outputs["Denoising Albedo"].links[0])
    bpy.ops.render.render(write_still=True)
    # Read render
    file_name = os.listdir(osp.join(output_dir, mask_path, str(day)))[0]
    image = cv2.imread(osp.join(output_dir, mask_path, str(day), file_name), cv2.IMREAD_UNCHANGED)
    
    # Compute number of white pixels (255)
    cnt = np.sum(image == 255) // 3
    
    # Reconnect normal and anomalous nodes
    node_tree.links.new(render_layers_node.outputs["Noisy Image"],
                        denoise_node.inputs["Image"])
    node_tree.links.new(render_layers_node.outputs["Denoising Normal"],
                        denoise_node.inputs["Normal"])
    node_tree.links.new(render_layers_node.outputs["Denoising Albedo"],
                        denoise_node.inputs["Albedo"])
    return cnt > min_pixels


def render_images(cameras, day, combinations, output_dir, anomaly_list, anomalies_dir, minpix, anomalous=False):
    '''iterate through cameras around station and render images
    
    Args:
        - cameras: blender dictionary of all cameras in the scene
        - day: current iteration of ephemeris data
        - combinations: list of scene parameters to be varied
        - output_dir: experiment number folder where camera renders are saved
        - anomaly_list: list of possible anomalous objects 
        - anomalies_dir: path to anomaly models
        - anomalous: boolean to make anomalous (True) or normal (False) images
        - minpix: minimum pixels of anomaly 
    
    '''
    # define objects in scene and setup file output
    sun_light = bpy.data.objects['sun light']
    station = bpy.data.objects['ISS']
    file_output_node = bpy.data.scenes["Scene"].node_tree.nodes["File Output"]
    file_output_node.base_path = output_dir + "/"
    bpy.context.scene.render.filepath = output_dir + "/" #"/trash/"
    # output node slots are folder names where images are saved
    fslots = file_output_node.file_slots
    anomaly_path = "anomaly"
    fb_mask_path = "fb_mask"
    anomaly_mask_path = "anomaly_mask"
    normal_path = "normal"

    # iterate through cameras in scene
    for cam in cameras:
        bpy.context.scene.camera = cam
        cam_render_path = osp.join(output_dir, cam.name)
        file_output_node.base_path = cam_render_path + "/"
        
        # set non-anomaly objects to pass index of 1 for foreground/background mask
        obj_list = ['ISS','moon','earth','sun']
        for i, obj in enumerate(obj_list):
            bpy.data.objects[obj].pass_index = 1
        
        if anomalous:
            # setup node tree for file saving folders
            fslots[0].path = str(osp.join(anomaly_path, str(day))) + "/"
            fslots[1].path = str(osp.join(fb_mask_path, str(day))) + "/"
            fslots[2].path = str(osp.join(anomaly_mask_path, str(day))) + "/"
            
            # add anomaly to the scene and set pass index to 2 for anomaly mask
            anomaly = load_anomaly(anomaly_list, anomalies_dir)
            anomaly_obj = bpy.data.objects[anomaly]
            anomaly_obj.pass_index = 2
            anomaly_obj.hide_render = False
            pixel_valid = False
            
            # iterate through scene parameter combinations
            for i, comb in enumerate(combinations):
                illum, depth, scale, colour = comb

                # Set scene parameter values
                sun_light.data.energy = illum   #* illumination
                if scale != -1: #* scale
                    set_anomaly_scale(anomaly_obj, scale)
                if i == 0:  # first combination must be depth=0 and default colour
                    assert depth == 0, "Relative depth must be 0 for first render"
                    assert colour == "default", "Colour must be 'default' for first render"
                    orig_colour = None
                    
                    #* depth
                    anomaly_pos = set_anomaly_position(anomaly_obj, station, cam)
                    if anomaly_pos is None:
                        break
                    # Check if anomaly has more than min pixels in render
                    bpy.data.scenes["Scene"].cycles.samples = 20
                    pixel_valid = check_anomaly_pixels(cam_render_path, anomaly_mask_path, day, minpix)
                    k = 0
                    while not pixel_valid and k < 10:
                        anomaly_pos = set_anomaly_position(anomaly_obj, station, cam)
                        pixel_valid = check_anomaly_pixels(cam_render_path, anomaly_mask_path, day, minpix)
                        k += 1
                    # if anomaly cannot meet pixel requirements move on to next combination
                    if not pixel_valid:
                        continue
                else:  # not first combination,  set anomaly 
                    pos_valid = set_anomaly_position(anomaly_obj, station, cam, depth, anomaly_pos)
                    # anomaly size is verified in subsequent processing script
                    if pos_valid is None:
                        continue
                    
                    # Set colour of anomaly
                    if colour == "default":
                        # Reset colour
                        if orig_colour is not None:
                            set_anomaly_colour(anomaly_obj, rgb=orig_colour)
                    elif colour in COLOURS:
                        # Modify colours
                        prev_colour = set_anomaly_colour(anomaly_obj, rgb=COLOURS[colour])
                        orig_colour = prev_colour if orig_colour is None else orig_colour
                    else:
                        continue
                
                # file name prefix lists scene parameters for this combination
                file_prefix = f"{anomaly}_{illum}_{scale}_{depth}_{colour}"
                bpy.data.scenes["Scene"].cycles.samples = 256
                render_anomaly_single(cam_render_path, file_prefix, anomaly_path, fb_mask_path, anomaly_mask_path, day)
            
            # Remove anomaly from scene
            bpy.data.objects[anomaly].select_set(True)
            bpy.ops.object.delete()
        else:
            # setup node tree
            fslots[0].path = str(osp.join(normal_path, str(day))) + "/"
            fslots[1].path = str(osp.join(fb_mask_path, str(day))) + "/"
            fslots[2].path = "/trash/"
            #? Illumination variation only
            for i, comb in enumerate(combinations):
                sun_light.data.energy = comb[0]
                file_prefix = f"normal_{comb[0]}"
                render_normal_single(cam_render_path, file_prefix, normal_path, fb_mask_path, day)
            # delete extra trash folder
            shutil.rmtree(cam_render_path + "/trash/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_num", type=int, required=False, default=-1)
    parser.add_argument("--anomaly", action="store_true", default=False, help="if true, will add anomaly to scene")
    parser.add_argument("--mode", nargs="*", default=[], help="modes of experiment")
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--config", type=str, required=False, default="/home/Blender-Render/render_config.yaml", help="path to config file")
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    return args


def render(args, cfg):
    ''' iterate through all days of ephemeris data and set up scene
    
    Args:
        - args: parsed arguments from command line
        - cfg: config params from main function
    '''
    print(f"Experiment: {args.exp_num}")
    
    # setup scene with device and render settings
    setup()
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # load object models, cameras/spotlights, and ephemeris data
    obj_list = ['ISS','moon','earth','sun']
    load_models(obj_list, cfg["cad_models_path"])
    objs = bpy.data.objects
    setup_cams(len(cfg["cams"]))
    ephemeris = load_ephemeris(cfg['ephemeris_path'])
    # get possible scene parameters
    sun_strength = cfg.get("sun_strength", [])
    depths = cfg.get("depths", [])
    scales = cfg.get("scales", [])
    colours = cfg.get("colours", [])
    # add and setup sun light
    default_sun_strength = 10
    setup_sunlight(default_sun_strength, objs['ISS'])
    
    # Iterate through days in ephemeris and render
    days = list(range(cfg["start_day"], cfg["end_day"]+1, cfg["day_interval"]))
    for day in tqdm(days):
        # set dimensions and locations of objects
        moon_radius = ephemeris["moon"][day, 0]
        moon_pos = ephemeris["moon"][day, 1:]
        objs["moon"].dimensions = (2*moon_radius,2*moon_radius,2*moon_radius)
        objs["moon"].location = moon_pos
        
        objs["ISS"].scale = (1,1,1)
        objs["ISS"].location = (0, 0, 0)
        
        earth_radius = ephemeris["earth"][day, 0]
        earth_pos = ephemeris["earth"][day, 1:]
        objs["earth"].dimensions = (2*earth_radius,2*earth_radius,2*earth_radius)
        objs["earth"].location = earth_pos
        
        sun_radius = ephemeris["sun"][day, 0]
        sun_pos = ephemeris["sun"][day, 1:]
        objs["sun"].dimensions = (2*sun_radius,2*sun_radius,2*sun_radius)
        objs["sun"].location = sun_pos + 150
        
        # set sun light position
        objs["sun"].location = sun_pos
        
        # setup cameras
        cam_positions = get_cam_pos(cfg["cams"], args.anomaly)
        cam_objs = [obj for obj in objs if obj.type == 'CAMERA']
        for i, cam in enumerate(cam_objs):
            if cam.name == "Camera":
                raise Exception("Default camera not removed")
            cam.name = "Camera" + str(int(cam_positions[i,0]))
            
            # Add random noise to camera position and orientation
            cam.location = cam_positions[i,1:4] + np.random.normal(0,1,3)
            cam.rotation_euler = cam_positions[i,4:7] + np.random.normal(0,0.2,3)
            
            # Setup spotlights to match camera
            try:
                spot = objs[f"Spot{int(cam_positions[i,0])}"]
            except:
                # Default name hasn't changed
                spot = objs['Spot_'+str(i+1)]
                spot.name = f"Spot{int(cam_positions[i,0])}"
            spot.location = cam.location
            spot.rotation_euler = cam.rotation_euler
        
        # Experiment details
        exp_dir = osp.join(cfg["output_dir"], f"exp_{args.exp_num}")  # experiment renders output folder
        # create list of possible scene parameter combinations
        options = OrderedDict()
        options["illumination"] = sun_strength if "illumination" in args.mode else [default_sun_strength]
        if args.anomaly:
            options["depths"] = depths if "depth" in args.mode else [0]
            options["scales"] = np.random.choice(scales, 1).tolist() \
                                    if "scale" in args.mode else [-1]
            options["colours"] = ["default"] + np.random.choice(colours, 1).tolist() \
                                if "colour" in args.mode else ["default"]
        
        opt_combs = list(it.product(*options.values()))
        print(opt_combs)

        # Render images
        render_images(cam_objs, day, opt_combs, exp_dir, cfg["anomalies"], cfg["anomalies_path"], cfg["min_pixel"], anomalous=args.anomaly)

if __name__ == "__main__":
    args = parse_args()
    config_file = args.config
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # If the 'cams' is defined with 'start' and 'end', convert it to a range
    if 'cams' in config and isinstance(config['cams'], dict):
        start = config['cams']['start']
        end = config['cams']['end']
        config['cams'] = list(range(start, end + 1))  
        
    render(args, config)