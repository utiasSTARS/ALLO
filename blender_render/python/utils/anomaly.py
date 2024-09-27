import bpy
import bpy_extras
import os
import os.path as osp
import numpy as np
import random
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .camera import get_camera_forward


def load_anomaly(anomaly_list, anomalies_dir):
    ''' add anomaly to scene 
    Args:
        - anomaly list: string list of possible anomaly files to load
        - anomalies_dir: path to folder where anomaly models are saved
    Returns:
        - anomaly_obj: string of anomaly name'''
    # randomly select an anomaly and load it in scene
    anomaly_obj = random.choice(anomaly_list)
    obj_path = osp.join(anomalies_dir, anomaly_obj + ".blend")
    bpy.ops.wm.append(filepath=osp.join(obj_path, 'Object', anomaly_obj),
                        directory=osp.join(obj_path, 'Object'), 
                        filename=anomaly_obj)
    return anomaly_obj

def loc_check(station, anomaly, cam):
    """Check if anomaly is in a valid position (camera can see it and it doesn't overlap with station)
    Args:
        - station: station model object
        - anomaly: anomaly model object
        - cam: camera object
    Returns: 
        -valid: booloan, False if location is good and True if location is bad
    """
    # 1. check if the anomaly goes through the station mesh
    # Get their world matrix
    mat1 = station.matrix_world
    mat2 = anomaly.matrix_world
    # Get the geometry in world coordinates
    vert1 = [mat1 @ v.co for v in station.data.vertices] 
    poly1 = [p.vertices for p in station.data.polygons]
    vert2 = [mat2 @ v.co for v in anomaly.data.vertices] 
    poly2 = [p.vertices for p in anomaly.data.polygons]
    # Create the BVH trees
    bvh1 = BVHTree.FromPolygons( vert1, poly1 )
    bvh2 = BVHTree.FromPolygons( vert2, poly2 )
    
    overlap = bvh1.overlap( bvh2 )  #* boolean for overlap
    
    # 2. check if anomaly is completely inside ISS mesh
    anomaly_location = anomaly.location
    x,y,z = anomaly_location
    dest_point = Vector((x, y, z+100)) # destination will be point far from the anomaly and station
    # create vector from station, through anomaly, to far away destination point
    origin =  station.matrix_local.inverted()@anomaly_location
    destination = station.matrix_local.inverted()@dest_point
    direction = (destination - origin).normalized() 
    # shooting ray from origin along destination point vector, if it hits then the object is sitting inside the station
    iss_hit, _, _, _ = station.ray_cast(origin, direction)  #* boolean for anomaly inside ISS 
    
    # 3. check if anomaly is occluded from the camera by the station
    cam_point = cam.location
    cam_destination = station.matrix_local.inverted()@cam_point
    cam_direction = (cam_destination - origin).normalized() #normalizing 
    # shooting ray from camera to the anomaly, if it hits the station then the object is hidden the station
    cam_hit, _, _, _ = station.ray_cast(origin, cam_direction)  #* boolean for anomaly hidden by ISS
    
    # 4. check if if anomaly is in view frustum 
    anom2D = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, anomaly_location)
    if anom2D.x > 0 and anom2D.x < 1 and anom2D.y > 0 and anom2D.y < 1:
        # inside frustum
        outside_frustum = False
    else:
        # outside frustum
        outside_frustum = True
    
    #* if anomaly inside or overlaping or too far with station mesh return True
    if any([iss_hit, overlap, cam_hit, outside_frustum]):
        valid = False
    else:  # anomaly can be seen by camera and doesn't overlap with station 
        valid = True
        
    return valid


def anomaly_box(cam_pos, cam_euler):
    """Calculates bounding box of possible anomaly locations
    Args:
        - cam_pos: position of camera
        - cam_euler: euler rotation of camera in degrees
    Returns:
        - bbox: bounding box
        - forward: camera forward vector
    """
    d = 4   #* distance from camera to approx box from
    ratio = 0.3    #* ratio of distance from camera to plane and box
    
    #* camera rotation matrix and forward vector
    forward = get_camera_forward(cam_euler)
    plane = np.array(cam_pos) + d*forward
    
    bbox = [plane[0]-ratio*d, plane[0]+ratio*d,
            plane[1]-ratio*d, plane[1]+ratio*d,
            plane[2]-ratio*d, plane[2]+ratio*d]
    return bbox, forward


def set_anomaly_position(anomaly, station, cam, d=1, prev_loc=None):
    ''' place anomaly inside camera frustrum
    Args:
        - anomaly: anomaly model object
        - station: station model object
        - cam: active camera
        - d: depth of anomaly from scene parameter combination
        - prev_loc: previous location of anomaly from past parameter combination
    Returns:
        - loc: new anomaly position or None if anomaly could not be placed in a valid location'''
    # if there is a previous location inputted (this isn't the first combination) only move anomaly along z axis relative to camera
    vary_z_only = prev_loc is not None
    if not vary_z_only:  # randomly rotate anomaly
        anomaly.rotation_euler = np.random.uniform(low=0, high=2*np.pi, size=(3,))
    bbox, forward = anomaly_box(cam.location, cam.rotation_euler)
    rng = np.random.default_rng()
    
    # place anomaly inside bounding box while checking validity of anomaly location
    valid = False
    i = 0
    while not valid and i < 100:
        if vary_z_only:  # move anomaly closer or further from camera
            loc = prev_loc + d * forward
        else:
            # set location of anomaly by uniform sampling in bounding box
            loc = rng.uniform([bbox[0], bbox[2], bbox[4]], 
                            [bbox[1], bbox[3], bbox[5]])
        anomaly.location = loc
        # update scene and check anomaly position 
        bpy.context.view_layer.update()
        valid = loc_check(station, anomaly, cam)
        i += 1
    if not valid:
        return None
    return loc
        
def set_anomaly_scale(anomaly, scale):
    """ Sets anomaly scale to specified value """
    anomaly.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    
    
def set_anomaly_colour(anomaly, rgb=None):
    """ set anomaly colour to specific rgb value
    Args:
        - anomaly: anomaly model object
        - rgb: 4-tuple colour code
    Returns:
        - mat_prev_colours: """
    #* Select object and select principled bsdf and change base colour
    mat_prev_colours = {}
    for mat in anomaly.data.materials.values():
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                assert rgb is not None, "Must provide rgb value"
                mat_prev_colours[mat.name] = node.inputs[0].default_value[:]
                if rgb is not None:
                    if isinstance(rgb, tuple):
                        node.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1)
                    elif isinstance(rgb, dict):
                        #? rgb is a dict of material names and rgb values
                        node.inputs[0].default_value = rgb[mat.name]
                    break
    bpy.context.view_layer.update()
    
    return mat_prev_colours