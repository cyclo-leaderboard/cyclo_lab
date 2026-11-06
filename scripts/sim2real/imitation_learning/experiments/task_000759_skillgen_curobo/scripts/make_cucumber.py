"""Approximate slightly flattened cucumber: stable convex geometry, no artificial attachment."""
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
path='/experiment/task_000759/cucumber.usda'
mesh=trimesh.creation.icosphere(subdivisions=2)
mesh.vertices*=np.array([.118,.021,.021])
mesh.vertices[:,2]=np.maximum(mesh.vertices[:,2],-.018)
mesh=mesh.convex_hull
stage=Usd.Stage.CreateNew(path);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
root=UsdGeom.Xform.Define(stage,'/Cucumber');stage.SetDefaultPrim(root.GetPrim())
m=UsdGeom.Mesh.Define(stage,'/Cucumber/geometry');m.CreatePointsAttr(mesh.vertices.tolist());m.CreateFaceVertexCountsAttr([3]*len(mesh.faces));m.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist());m.CreateSubdivisionSchemeAttr('none')
UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
mat=UsdShade.Material.Define(stage,'/Cucumber/visual');shader=UsdShade.Shader.Define(stage,'/Cucumber/visual/shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08,.31,.025));shader.CreateOutput('surface',Sdf.ValueTypeNames.Token);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.12)
stage.GetRootLayer().Save();print(path,len(mesh.vertices),len(mesh.faces))
