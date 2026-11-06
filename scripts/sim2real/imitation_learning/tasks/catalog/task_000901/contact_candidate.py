"""Episode-local contact hypotheses; no measured rubber constants claimed."""
import json,re
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema

def apply_candidate(stage, robot_path, mode):
 assert mode in ('baseline','recess1mm','recess3mm','recess5mm'),mode
 recess_m={'baseline':0.,'recess1mm':.001,'recess3mm':.003,'recess5mm':.005}[mode]
 material=stage.GetPrimAtPath(robot_path+'/distalGripperPhysicsMaterial')
 api=PhysxSchema.PhysxMaterialAPI.Apply(material)
 stiffness=0.
 damping=0.
 api.CreateCompliantContactStiffnessAttr().Set(stiffness)
 api.CreateCompliantContactDampingAttr().Set(damping)
 api.CreateCompliantContactAccelerationSpringAttr().Set(False)
 report={'mode':mode,'stiffness_N_per_m':stiffness,'damping_Ns_per_m':damping,'coefficients_measured':False,'recess_max_m':recess_m,'visual_mesh_changed':False,'contact_model':'Rigid collision; no rubber compliance','pieces':[]}
 groups={};cache=UsdGeom.XformCache()
 for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
  path=str(prim.GetPath());m=re.search(r'/gripper_([lr])_[^/]*_([lr]2)/',path)
  if m and '_cyclo_softgripper_proxy/' in path and prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
   if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
   mesh=UsdGeom.Mesh(prim);pts=np.asarray(mesh.GetPointsAttr().Get(),float);tf=cache.GetLocalToWorldTransform(prim)
   world=np.array([tf.Transform(Gf.Vec3d(*v)) for v in pts]);groups.setdefault(m.groups(),[]).append((prim,mesh,pts,world,tf))
 assert len(groups)==4 and all(len(x)==5 for x in groups.values())
 centers={key:np.concatenate([x[3] for x in group]).mean(axis=0) for key,group in groups.items()}
 for (side,finger),group in groups.items():
  all_world=np.concatenate([item[3] for item in group]);center=all_world.mean(axis=0)
  _,_,axes=np.linalg.svd(all_world-center,full_matrices=False);long_axis=axes[0]
  longitudinal=all_world@long_axis;long_min,long_max=float(longitudinal.min()),float(longitudinal.max())
  other=(side,'r2' if finger=='l2' else 'l2');inward=centers[other]-centers[(side,finger)]
  inward-=np.dot(inward,long_axis)*long_axis;inward/=np.linalg.norm(inward)
  for prim,mesh,pts,world,tf in group:
   proj=world@inward;span=np.ptp(proj);assert span>.001
   # Smoothly shave the inner half: outer surface unchanged, innermost 1 mm recessed.
   weight=np.clip((proj-proj.min())/span*2-1,0,1)
   weight=weight*weight*(3-2*weight)
   length_fraction=np.clip((world@long_axis-long_min)/(long_max-long_min),0,1)
   arch=np.sin(np.pi*length_fraction)**2
   shift=recess_m*weight*arch
   moved=world-shift[:,None]*inward
   inv=tf.GetInverse();local=np.array([inv.Transform(Gf.Vec3d(*v)) for v in moved])
   if recess_m:
    mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in local]);mesh.GetExtentAttr().Set([Gf.Vec3f(*local.min(axis=0)),Gf.Vec3f(*local.max(axis=0))])
   report['pieces'].append({'path':str(prim.GetPath()),'vertex_count':len(pts),'inward_world_direction':inward.tolist(),'max_world_shift_m':float(shift.max()),'min_world_shift_m':float(shift.min()),'local_bounds_before':[pts.min(axis=0).tolist(),pts.max(axis=0).tolist()],'world_before':world.tolist(),'world_after':moved.tolist(),'longitudinal_axis':long_axis.tolist(),'longitudinal_min_max':[long_min,long_max],'shape_note':'Rounded longitudinal pocket: sin(pi*s)^2 recession, zero at both ends, smooth inner-half taper; per-band convex hull approximation. No longitudinal displacement.'})
 return report
