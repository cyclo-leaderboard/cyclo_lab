"""Generate the fixed-size white table USD used by WhiteRoom."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


WIDTH = 1.2
DEPTH = 0.597
TOP_HEIGHT = 0.685
TOP_THICKNESS = 0.03
CORNER_RADIUS = 0.04
LEG_WIDTH = 0.05
LEG_EDGE_INSET = 0.035
ARC_SEGMENTS = 8


def make_top_outline():
    half_width = WIDTH / 2
    half_depth = DEPTH / 2
    corners = (
        (half_width - CORNER_RADIUS, half_depth - CORNER_RADIUS, 0),
        (-half_width + CORNER_RADIUS, half_depth - CORNER_RADIUS, 90),
        (-half_width + CORNER_RADIUS, -half_depth + CORNER_RADIUS, 180),
        (half_width - CORNER_RADIUS, -half_depth + CORNER_RADIUS, 270),
    )
    return [
        (
            cx + CORNER_RADIUS * math.cos(math.radians(start + step * 90 / ARC_SEGMENTS)),
            cy + CORNER_RADIUS * math.sin(math.radians(start + step * 90 / ARC_SEGMENTS)),
        )
        for cx, cy, start in corners
        for step in range(ARC_SEGMENTS + 1)
    ]


def main():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    table = UsdGeom.Xform.Define(stage, "/WhiteTable")
    stage.SetDefaultPrim(table.GetPrim())

    material = UsdShade.Material.Define(stage, "/WhiteTable/Looks/WhitePaint")
    shader = UsdShade.Shader.Define(stage, "/WhiteTable/Looks/WhitePaint/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.833, 0.82875, 0.82025))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(material)

    top_material = UsdShade.Material.Define(stage, "/WhiteTable/Looks/WhiteLaminateTop")
    top_shader = UsdShade.Shader.Define(stage, "/WhiteTable/Looks/WhiteLaminateTop/PreviewSurface")
    top_shader.CreateIdAttr("UsdPreviewSurface")
    top_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    top_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.16)
    texture = UsdShade.Shader.Define(stage, "/WhiteTable/Looks/WhiteLaminateTop/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/white_table_laminate.png"))
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.85, 0.85, 0.85, 1.0))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    primvar_reader = UsdShade.Shader.Define(stage, "/WhiteTable/Looks/WhiteLaminateTop/PrimvarReader")
    primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
    primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    primvar_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(
        primvar_reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    top_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    top_material.CreateSurfaceOutput().ConnectToSource(top_shader.ConnectableAPI(), "surface")

    outline = make_top_outline()
    count = len(outline)
    bottom = TOP_HEIGHT - TOP_THICKNESS
    points = [Gf.Vec3f(x, y, TOP_HEIGHT) for x, y in outline]
    points += [Gf.Vec3f(x, y, bottom) for x, y in outline]
    faces = [list(range(count)), list(range(2 * count - 1, count - 1, -1))]
    faces += [
        [index, index + count, (index + 1) % count + count, (index + 1) % count]
        for index in range(count)
    ]
    top = UsdGeom.Mesh.Define(stage, "/WhiteTable/Top")
    top.CreatePointsAttr(points)
    top.CreateFaceVertexCountsAttr([len(face) for face in faces])
    top.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    top.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    top.CreateExtentAttr([
        Gf.Vec3f(-WIDTH / 2, -DEPTH / 2, bottom),
        Gf.Vec3f(WIDTH / 2, DEPTH / 2, TOP_HEIGHT),
    ])
    st = [
        Gf.Vec2f((x + WIDTH / 2) / WIDTH, (y + DEPTH / 2) / DEPTH)
        for x, y in outline
    ]
    UsdGeom.PrimvarsAPI(top).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(st + st)
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(top.GetPrim()).CreateApproximationAttr().Set("convexHull")
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(top_material)

    leg_height = bottom
    leg_x = WIDTH / 2 - LEG_EDGE_INSET - LEG_WIDTH / 2
    leg_y = DEPTH / 2 - LEG_EDGE_INSET - LEG_WIDTH / 2
    for x_name, x in (("Left", -leg_x), ("Right", leg_x)):
        for y_name, y in (("Front", -leg_y), ("Back", leg_y)):
            leg = UsdGeom.Cube.Define(stage, f"/WhiteTable/Leg{y_name}{x_name}")
            leg.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(leg.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, leg_height / 2))
            xform.AddScaleOp().Set(Gf.Vec3f(LEG_WIDTH, LEG_WIDTH, leg_height))
            UsdPhysics.CollisionAPI.Apply(leg.GetPrim())

    output = Path(__file__).with_name("white_table.usda")
    if not stage.Export(str(output)):
        raise RuntimeError(f"Could not export {output}")


if __name__ == "__main__":
    main()
