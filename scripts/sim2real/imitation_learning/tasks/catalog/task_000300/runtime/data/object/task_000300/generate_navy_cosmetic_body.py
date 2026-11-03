"""Regenerate only the gently rounded Task000300 bottle body in its USD asset."""

from __future__ import annotations

import math
from pathlib import Path


ASSET = Path(__file__).with_name("navy_cosmetic.usda")
HALF_WIDTH_M = 0.053 / 2
BOTTOM_Z_M = -0.0725
TOP_Z_M = 0.0365
SUPERELLIPSE_POWER = 5.0
PERIMETER_SAMPLES = 64


def fmt_vec(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{value:.8f}" for value in values) + ")"


def fmt_array(values: list[object], indent: str = "        ") -> str:
    return "[\n" + "".join(f"{indent}    {value},\n" for value in values) + indent + "]"


def main() -> None:
    text = ASSET.read_text(encoding="utf-8")
    material_marker = '        # Four gently rounded side walls; the lid retains NavyPlastic.\n'
    if material_marker not in text:
        material = '''        # Four gently rounded side walls; the lid retains NavyPlastic.
        def Material "GlossyBodyPlastic"
        {
            token outputs:surface.connect = </NavyCosmetic/Looks/GlossyBodyPlastic/Shader.outputs:surface>
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.06, 0.085, 0.14)
                float inputs:ior = 1.75
                float inputs:metallic = 0.01
                float inputs:roughness = 0.025
                float inputs:clearcoat = 1.0
                float inputs:clearcoatRoughness = 0.005
                token outputs:surface
            }
        }

'''
        # A material at the root is intentionally retained for the unmodified lid.
        text = text.replace('    }\n\n    # Square body:', material + '    }\n\n    # Square body:', 1)

    body_marker = (
        '    # Square body:' if '    # Square body:' in text
        else '    # 53 x 53 x 109 mm convex body'
    )
    start = text.index(body_marker)
    end = text.index('    # Round lid:', start)

    perimeter: list[tuple[float, float]] = []
    side_normals: list[tuple[float, float, float]] = []
    exponent = 2.0 / SUPERELLIPSE_POWER
    for i in range(PERIMETER_SAMPLES):
        angle = 2 * math.pi * i / PERIMETER_SAMPLES
        cosine, sine = math.cos(angle), math.sin(angle)
        x = HALF_WIDTH_M * math.copysign(abs(cosine) ** exponent, cosine)
        y = HALF_WIDTH_M * math.copysign(abs(sine) ** exponent, sine)
        perimeter.append((x, y))
        nx = math.copysign(abs(x / HALF_WIDTH_M) ** (SUPERELLIPSE_POWER - 1), x)
        ny = math.copysign(abs(y / HALF_WIDTH_M) ** (SUPERELLIPSE_POWER - 1), y)
        length = math.hypot(nx, ny)
        side_normals.append((nx / length, ny / length, 0.0))

    n = PERIMETER_SAMPLES
    points = [(x, y, BOTTOM_Z_M) for x, y in perimeter]
    points += [(x, y, TOP_Z_M) for x, y in perimeter]
    face_counts = [n, n] + [4] * n
    # The perimeter is counterclockwise when viewed from above.
    face_indices = list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
    normals = [(0.0, 0.0, -1.0)] * n + [(0.0, 0.0, 1.0)] * n
    for i in range(n):
        j = (i + 1) % n
        face_indices.extend((i, j, n + j, n + i))
        normals.extend((side_normals[i], side_normals[j], side_normals[j], side_normals[i]))

    body = f'''    # 53 x 53 x 109 mm convex body with four subtly curved sides.
    # Its lower and upper planes stay at the original heights for stable placement.
    def Mesh "Body" (
        apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding = </NavyCosmetic/Looks/GlossyBodyPlastic>
        bool physics:collisionEnabled = true
        uniform token physics:approximation = "convexHull"
        point3f[] points = {fmt_array([fmt_vec(point) for point in points])}
        int[] faceVertexCounts = {fmt_array(face_counts)}
        int[] faceVertexIndices = {fmt_array(face_indices)}
        normal3f[] normals = {fmt_array([fmt_vec(normal) for normal in normals])} (
            interpolation = "faceVarying"
        )
        uniform token subdivisionScheme = "none"
    }}

'''
    ASSET.write_text(text[:start] + body + text[end:], encoding="utf-8")


if __name__ == "__main__":
    main()
