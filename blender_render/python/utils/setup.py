import bpy
import os

def setup():
    ''' setup blender scene, render engine, and render resolution '''
    # delete everything currently in file
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # render engine settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
    
    print("engine: " + bpy.context.scene.render.engine)
    print("device: " + bpy.context.scene.cycles.device)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print(d["name"], d["use"])
    
    # render output resolution settings
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    print("resolution x: " + str(bpy.context.scene.render.resolution_x))
    print("resolution y: " + str(bpy.context.scene.render.resolution_y))
        
def load_models(obj_list, cad_model_path):
    ''' load blender models of objects in list '''
    for obj in obj_list:
        file_path = cad_model_path + obj + '.blend/'
        inner_path = 'Object'
        object_name = obj
        bpy.ops.wm.append(filepath=os.path.join(file_path, inner_path, object_name),
                          directory=os.path.join(file_path, inner_path),
                          filename=object_name)

def setup_cams(n):
    """ add n cameras and spotlights to the blender scene at origin
    Args:
        - n (int): number of cameras that render
    """
    #? setup spotlights and camera
    for cam_num in range(n):
        #* cameras
        bpy.ops.object.camera_add(
            enter_editmode=False, align='VIEW', 
            location=(0, 0, 0), 
            rotation=(0, 0, 0), 
            scale=(1, 1, 1)
        )
        bpy.data.objects["Camera"].data.clip_end = 10000
        bpy.data.objects["Camera"].data.lens = 25
        bpy.data.objects["Camera"].name = "Camera_"+str(cam_num+1)
        
        #* spotlights
        bpy.ops.object.light_add(
            type='SPOT', radius=10.0, align='VIEW',
            location=(0, 0, 0), 
            rotation=(0, 0, 0), 
            scale=(1, 1, 1)
        )
        bpy.data.objects["Spot"].data.energy = 500
        bpy.data.objects["Spot"].data.spot_size = 80
        bpy.data.objects["Spot"].name = "Spot_"+str(cam_num+1)
