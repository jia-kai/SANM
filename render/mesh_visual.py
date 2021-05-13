import bpy

import sys
import json
import os
from typing import List, Tuple
from pathlib import Path
import math

g_basedir = Path(__file__).resolve().parent
sys.path.append(str(g_basedir / 'blender-cli-rendering'))
import utils

def set_scene_objects(config, input_path) -> bpy.types.Object:
    # Instantiate a floor plane
    # utils.create_plane(size=200.0, location=(0.0, 0.0, -1.0))

    # Instantiate a triangle mesh
    bpy.ops.import_scene.obj(filepath=input_path)
    current_object, = bpy.context.selected_objects[:]
    for i in config['transforms']:
        name = i.pop('name')
        if name == 'abs_rotation_deg':
            current_object.rotation_euler = list(map(math.radians, i['value']))
        elif name == 'rotate':
            # I don't know why this fails on blender 2.92 (error: context is
            # incorrect). Here is a dirty fix
            assert len(i) == 2
            setattr(current_object.rotation_euler,
                    i['orient_axis'].lower(), i['value'])
        else:
            func = getattr(bpy.ops.transform, name)
            func(**i)

    if config.get('smooth'):
        for f in current_object.data.polygons:
            f.use_smooth = True

    # Setup a material with wireframe visualization and per-face colors
    mat = utils.add_material("Material_Visualization", use_nodes=True,
                             make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    r, g, b = config['color']
    utils.set_principled_node(
        principled_node=principled_node,
        base_color=(r / 255, g / 255, b / 255, 1.0),
        metallic=0.5,
        specular=0.6,
        roughness=0.7,
    )

    tex_path = config.get('texture')
    if tex_path is not None:
        tex_img = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_img.image = bpy.data.images.load(str(g_basedir / tex_path))
        links.new(principled_node.inputs['Base Color'],
                  tex_img.outputs['Color'])

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    current_object.data.materials.clear()
    current_object.data.materials.append(mat)

    bpy.ops.object.empty_add(location=(0.0, 0.0, 0.0))
    focus_target = bpy.context.object

    return focus_target

def main():
    assert '--' in sys.argv, (
        f'usage: {sys.argv[0]} -- '
        'config resolution_ratio nr_sample input output')
    argv = sys.argv[sys.argv.index('--') + 1:]
    assert len(argv) == 5
    with open(argv[0]) as fin:
        config = json.load(fin)

    resolution_percentage = int(argv[1])
    num_samples = int(argv[2])
    input_path = argv[3]
    output_path = argv[4]

    # Parameters
    hdri_path = str(
        Path(__file__).resolve().parent / 'blender-cli-rendering' /
        'assets' / 'HDRIs' / 'green_point_park_2k.hdr'
    )

    # Scene Building
    scene = bpy.data.scenes["Scene"]
    world = scene.world

    ## Reset
    utils.clean_objects()

    ## Object
    focus_target_object = set_scene_objects(config, input_path)

    ## Camera
    camera_object = utils.create_camera(location=(.0, -5.0, 0.0))

    utils.add_track_to_constraint(camera_object, focus_target_object)
    utils.set_camera_params(camera_object.data, focus_target_object,
                            lens=72, fstop=1)

    ## Lights
    # utils.build_environment_texture_background(world, hdri_path)
    utils.create_area_light(location=(6.0, 0.0, 4.0),
                            rotation=(0.0, math.pi * 60.0 / 180.0, 0.0),
                            size=5.0,
                            color=(1.00, 0.70, 0.60, 1.00),
                            strength=1500.0,
                            name="Main Light")
    utils.create_area_light(location=(-6.0, 0.0, 2.0),
                            rotation=(0.0, -math.pi * 80.0 / 180.0, 0.0),
                            size=5.0,
                            color=(0.30, 0.42, 1.00, 1.00),
                            strength=1000.0,
                            name="Sub Light")
    bpy.ops.object.light_add(type='SUN', location=(0, -1, 5))
    obj = bpy.context.object
    light = obj.data
    light.energy = 5
    light.angle = math.pi * 10.0 / 180.0
    utils.add_track_to_constraint(obj, focus_target_object)

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    if not os.getenv('NO_BG'):
        # Render Setting
        utils.set_output_properties(scene, resolution_percentage, output_path)
        utils.set_cycles_renderer(scene, camera_object, num_samples,
                                  use_transparent_bg=True)



if __name__ == '__main__':
    main()
