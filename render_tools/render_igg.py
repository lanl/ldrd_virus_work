"""
Produce a nice render of PDB ID 1HZH--an example
of a Y-shaped human antibody with structure determined
at high resolution.
"""

import MDAnalysis as mda
from MDAnalysis import transformations
from ggmolvis.ggmolvis import GGMolVis
import bpy


def render():
    ggmv = GGMolVis()
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if 'Apple' in device.name and 'GPU' in device.name:
            device.use = True
        else:
            device.use = False

    u = mda.Universe("1hzh.pdb")
    system = u.select_atoms("all")
    workflow = [transformations.rotate.rotateby(160,
                                                direction=[1, 0, 0],
                                                ag=system),
                transformations.rotate.rotateby(30,
                                                direction=[0, 1, 0],
                                                ag=system),
                transformations.rotate.rotateby(95,
                                                direction=[0, 0, 1],
                                                ag=system),
                ]
    u.trajectory.add_transformations(*workflow)
    all_mol = ggmv.molecule(system, style="ribbon")
    all_mol.render(resolution=(2000, 2000),
                   filepath="1hzh.png",
                   lens=60,
                   composite_bg_rgba=(0.8, 0.8, 0.8, 1.0))

if __name__ == "__main__":
    render()
