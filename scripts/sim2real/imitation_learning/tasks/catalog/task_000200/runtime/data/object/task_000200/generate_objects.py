"""Generate the Task000200 plate and tape USDs without starting Isaac Sim.

The visible surfaces and the convex collision sectors use the same radial
profile.  Individual convex sectors keep the plate's hollow and tape's hole
open to PhysX, unlike a single convex hull around either complete object.
"""

from __future__ import annotations

import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLATE_FLAT_RADIUS = 0.075
PLATE_PROFILE = (
    # radius, underside z, eating-surface z (metres)
    (0.0, 0.0, 0.003),
    (PLATE_FLAT_RADIUS, 0.0, 0.003),
    (0.080, 0.003, 0.006),
    (0.086, 0.010, 0.014),
    (0.091, 0.016, 0.022),
    (0.095, 0.019, 0.024),
)
TAPE_INNER_RADIUS = 0.0375
TAPE_PAPER_OUTER_RADIUS = 0.0435
TAPE_OUTER_RADIUS = 0.054
TAPE_HEIGHT = 0.05
VISUAL_SIDES = 96
PLATE_COLLISION_SIDES = 32
TAPE_COLLISION_SIDES = 48


def _linear_hex(color: str) -> tuple[float, float, float]:
    """UsdPreviewSurface diffuseColor is linear; supplied hex swatches are sRGB."""
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    return tuple(value / 12.92 if value <= 0.04045 else
                 ((value + 0.055) / 1.055) ** 2.4 for value in channels)


def _point(radius: float, angle: float, z: float) -> tuple[float, float, float]:
    return (radius * math.cos(angle), radius * math.sin(angle), z)


def _mesh_text(name: str, points, faces, *, material=None, collider=False) -> str:
    def vectors(values):
        return ", ".join("(" + ", ".join(f"{value:.8f}" for value in row) + ")" for row in values)

    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    schemas = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]' if collider else '["MaterialBindingAPI"]'
    extra = (
        '        bool physics:collisionEnabled = true\n'
        '        uniform token physics:approximation = "convexHull"\n'
        '        token visibility = "invisible"\n'
        if collider else f'        rel material:binding = </{name.split("/")[0]}/Looks/{material}>\n'
    )
    prim_name = name.split("/")[-1]
    return (
        f'    def Mesh "{prim_name}" (\n        apiSchemas = {schemas}\n    )\n    {{\n'
        f"{extra}"
        f"        point3f[] points = [{vectors(points)}]\n"
        f"        int[] faceVertexCounts = [{counts}]\n"
        f"        int[] faceVertexIndices = [{indices}]\n"
        '        uniform token subdivisionScheme = "none"\n'
        "    }\n"
    )


def _ring_mesh(inner: float, outer: float, lower_inner: float, lower_outer: float,
               upper_inner: float, upper_outer: float, sides: int):
    """Closed annular volume, with the same radial bounds as its collision wedges."""
    points = []
    for radius, z in ((inner, lower_inner), (outer, lower_outer),
                      (inner, upper_inner), (outer, upper_outer)):
        points.extend(_point(radius, 2 * math.pi * i / sides, z) for i in range(sides))
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend((
            [i, j, sides + j, sides + i],
            [2 * sides + i, 3 * sides + i, 3 * sides + j, 2 * sides + j],
            [i, 2 * sides + i, 2 * sides + j, j],
            [sides + i, sides + j, 3 * sides + j, 3 * sides + i],
        ))
    return points, faces


def _wedge_mesh(inner: float, outer: float, lower_inner: float, lower_outer: float,
                upper_inner: float, upper_outer: float, angle0: float, angle1: float):
    points = [
        _point(radius, angle, z)
        for radius, z in ((inner, lower_inner), (outer, lower_outer),
                          (inner, upper_inner), (outer, upper_outer))
        for angle in (angle0, angle1)
    ]
    faces = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
    ]
    return points, faces


def _material(name: str, rgb: tuple[float, float, float], roughness: float,
              clearcoat: float = 0.0) -> str:
    return (
        f'        def Material "{name}"\n        {{\n'
        f'            token outputs:surface.connect = </ROOT/Looks/{name}/Shader.outputs:surface>\n'
        '            def Shader "Shader"\n            {\n'
        '                uniform token info:id = "UsdPreviewSurface"\n'
        f'                color3f inputs:diffuseColor = ({rgb[0]}, {rgb[1]}, {rgb[2]})\n'
        f'                float inputs:roughness = {roughness}\n'
        f'                float inputs:clearcoat = {clearcoat}\n'
        '                float inputs:clearcoatRoughness = 0.025\n'
        '                token outputs:surface\n            }\n        }\n'
    )


def _write_asset(root: str, materials, meshes, mass: float, output: Path):
    contents = (
        '#usda 1.0\n(\n    defaultPrim = "' + root + '"\n'
        '    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        f'def Xform "{root}" (\n'
        '    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n'
        '    bool physics:rigidBodyEnabled = true\n'
        f'    float physics:mass = {mass}\n'
        '    def Scope "Looks"\n    {\n'
        + "".join(_material(*material) for material in materials).replace('/ROOT/', f'/{root}/')
        + '    }\n'
        + "".join(meshes)
        + '}\n'
    )
    output.write_text(contents)


def generate_plate():
    root = "YellowPlate"
    meshes = []
    for band, ((inner, lower_inner, upper_inner),
               (outer, lower_outer, upper_outer)) in enumerate(zip(PLATE_PROFILE, PLATE_PROFILE[1:])):
        points, faces = _ring_mesh(inner if inner else 0.00001, outer,
                                   lower_inner, lower_outer, upper_inner, upper_outer,
                                   VISUAL_SIDES)
        meshes.append(_mesh_text(f"{root}/VisualBand{band}", points, faces, material="YellowPlastic"))
        for sector in range(PLATE_COLLISION_SIDES):
            angle0 = 2 * math.pi * sector / PLATE_COLLISION_SIDES
            angle1 = 2 * math.pi * (sector + 1) / PLATE_COLLISION_SIDES
            points, faces = _wedge_mesh(inner, outer, lower_inner, lower_outer,
                                         upper_inner, upper_outer, angle0, angle1)
            meshes.append(_mesh_text(f"{root}/ColliderBand{band}_{sector:02d}",
                                     points, faces, collider=True))
    _write_asset(root, [("YellowPlastic", _linear_hex("#ccd955"), 0.38, 0.12)], meshes,
                 0.058, HERE / "yellow_plate.usda")


def generate_tape():
    root = "BlueTape"
    meshes = []
    bands = (
        (TAPE_INNER_RADIUS, TAPE_PAPER_OUTER_RADIUS, "BrownPaper"),
        (TAPE_PAPER_OUTER_RADIUS, TAPE_OUTER_RADIUS, "BlueTape"),
    )
    for band, (inner, outer, material) in enumerate(bands):
        points, faces = _ring_mesh(inner, outer, 0.0, 0.0, TAPE_HEIGHT, TAPE_HEIGHT,
                                   VISUAL_SIDES)
        meshes.append(_mesh_text(f"{root}/VisualBand{band}", points, faces, material=material))
        if band == 0:
            # The cardboard's exposed top face is lighter than its inner wall.
            top_points = [
                _point(radius, 2 * math.pi * i / VISUAL_SIDES, TAPE_HEIGHT + 0.00002)
                for radius in (inner, outer)
                for i in range(VISUAL_SIDES)
            ]
            top_faces = [
                [i, VISUAL_SIDES + i, VISUAL_SIDES + (i + 1) % VISUAL_SIDES,
                 (i + 1) % VISUAL_SIDES]
                for i in range(VISUAL_SIDES)
            ]
            meshes.append(_mesh_text(f"{root}/PaperTop", top_points, top_faces,
                                     material="BrownPaperTop"))
        for sector in range(TAPE_COLLISION_SIDES):
            angle0 = 2 * math.pi * sector / TAPE_COLLISION_SIDES
            angle1 = 2 * math.pi * (sector + 1) / TAPE_COLLISION_SIDES
            points, faces = _wedge_mesh(inner, outer, 0.0, 0.0,
                                         TAPE_HEIGHT, TAPE_HEIGHT, angle0, angle1)
            meshes.append(_mesh_text(f"{root}/ColliderBand{band}_{sector:02d}",
                                     points, faces, collider=True))
    _write_asset(root, [
        ("BrownPaper", _linear_hex("#916a46"), 0.88, 0.0),
        ("BrownPaperTop", _linear_hex("#a18265"), 0.78, 0.0),
        ("BlueTape", _linear_hex("#180b2e"), 0.065, 1.0),
    ], meshes, 0.09, HERE / "blue_tape.usda")


if __name__ == "__main__":
    generate_plate()
    generate_tape()
