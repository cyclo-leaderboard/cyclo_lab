"""Build the Task000300 orange tray with 3 cm rounded corners and matching collision."""

from __future__ import annotations

import math
from pathlib import Path


OUTPUT = Path(__file__).with_name("orange_tray.usda")
HALF_X_M = 0.262 / 2
HALF_Y_M = 0.195 / 2
HEIGHT_M = 0.020
FLOOR_M = 0.006
WALL_M = 0.006
OUTER_RADIUS_M = 0.030
ARC_SEGMENTS_PER_CORNER = 6


def rounded_perimeter(radius: float) -> list[tuple[float, float]]:
    center_x = HALF_X_M - OUTER_RADIUS_M
    center_y = HALF_Y_M - OUTER_RADIUS_M
    corners = (
        (center_x, -center_y, -math.pi / 2),
        (center_x, center_y, 0.0),
        (-center_x, center_y, math.pi / 2),
        (-center_x, -center_y, math.pi),
    )
    return [
        (
            cx + radius * math.cos(start + step * math.pi / (2 * ARC_SEGMENTS_PER_CORNER)),
            cy + radius * math.sin(start + step * math.pi / (2 * ARC_SEGMENTS_PER_CORNER)),
        )
        for cx, cy, start in corners
        for step in range(ARC_SEGMENTS_PER_CORNER)
    ]


def point(x: float, y: float, z: float) -> str:
    return f"({x:.8f}, {y:.8f}, {z:.8f})"


def array(values: list[str | int]) -> str:
    return "[\n" + "".join(f"            {value},\n" for value in values) + "        ]"


def mesh(name: str, points: list[tuple[float, float, float]], faces: list[list[int]], *,
         collision: bool = False, invisible: bool = False) -> str:
    api = ' (\n        apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )' if collision else ""
    attrs = (
        '        bool physics:collisionEnabled = true\n'
        '        uniform token physics:approximation = "convexHull"\n'
        if collision else ""
    )
    if invisible:
        attrs += '        token visibility = "invisible"\n'
    return (
        f'    def Mesh "{name}"{api}\n    {{\n{attrs}'
        f'        point3f[] points = {array([point(*xyz) for xyz in points])}\n'
        f'        int[] faceVertexCounts = {array([len(face) for face in faces])}\n'
        f'        int[] faceVertexIndices = {array([index for face in faces for index in face])}\n'
        '        uniform token subdivisionScheme = "none"\n'
        '    }\n'
    )


def main() -> None:
    outer = rounded_perimeter(OUTER_RADIUS_M)
    inner = rounded_perimeter(OUTER_RADIUS_M - WALL_M)
    n = len(outer)

    bottom_points = [(x, y, z) for z in (0.0, FLOOR_M) for x, y in outer]
    bottom_faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    bottom_faces += [
        [i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)
    ]

    wall_points = [
        (x, y, z)
        for ring, z in ((outer, FLOOR_M), (outer, HEIGHT_M),
                        (inner, FLOOR_M), (inner, HEIGHT_M))
        for x, y in ring
    ]
    wall_faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        wall_faces.extend((
            [i, j, n + j, n + i],
            [2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j],
            [n + i, n + j, 3 * n + j, 3 * n + i],
            [i, 2 * n + i, 2 * n + j, j],
        ))

    colliders = []
    for i in range(n):
        j = (i + 1) % n
        footprint = [outer[i], outer[j], inner[j], inner[i]]
        points = [(x, y, z) for z in (FLOOR_M, HEIGHT_M) for x, y in footprint]
        faces = [
            [3, 2, 1, 0], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        colliders.append(mesh(f"WallCollider_{i:02d}", points, faces,
                              collision=True, invisible=True))

    header = '''#usda 1.0
(
    defaultPrim = "WhiteBasket"
    metersPerUnit = 1
    upAxis = "Z"
)

# Outer size: 26.2 x 19.5 x 2.0 cm; 0.6 cm floor/walls; 3 cm outer corner radius.
def Xform "WhiteBasket" (
    apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    rel material:binding = </WhiteBasket/Looks/OrangePlastic>
    bool physics:rigidBodyEnabled = true
    float physics:mass = 0.13

    def Scope "Looks"
    {
        def Material "OrangePlastic"
        {
            token outputs:surface.connect = </WhiteBasket/Looks/OrangePlastic/Shader.outputs:surface>
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                # Linear RGB equivalent of sRGB #ff6419, selected from the
                # Task000300 real-camera tray pixels.
                color3f inputs:diffuseColor = (1.000000, 0.127438, 0.009721)
                float inputs:metallic = 0
                float inputs:roughness = 0.35
                token outputs:surface
            }
        }
    }

'''
    body = (
        mesh("Bottom", bottom_points, bottom_faces, collision=True)
        + "\n"
        + mesh("Walls", wall_points, wall_faces)
        + "\n"
        + "\n".join(colliders)
    )
    OUTPUT.write_text(header + body + "}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
