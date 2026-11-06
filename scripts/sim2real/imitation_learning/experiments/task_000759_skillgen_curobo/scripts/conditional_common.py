"""Shared isolated scene and explicit optimization profiles. No application startup."""
import numpy as np
from curobo.util_file import load_yaml
from plan_goalset import BASE
from plan_pickplace import prior_spline

def static_world(scene,fk):
    w={'table':dict(dims=[*scene['table_size_xy'],.05],pose=[*scene['table_center_xy'],scene['table_top_m']-.025,1,0,0,0]),'torso':dict(dims=[.25,.30,.55],pose=[-.045,0,.88,1,0,0,0])}
    for i in range(2,8):w[f'left_{i}']=dict(dims=[.11]*3,pose=[*fk['body_pose'][0,list(fk['body_names']).index(f'arm_l_link{i}'),:3].tolist(),1,0,0,0])
    b=scene['basket'];bx,by=b['center_xy'];sx,sy=b['inner_size_xy'];t=b['wall_thickness_m'];h=b['wall_top_m']-b['floor_top_m']
    w['floor']=dict(dims=[sx+2*t,sy+2*t,.02],pose=[bx,by,b['floor_top_m']-.01,1,0,0,0])
    for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
        pos=[bx,by,b['floor_top_m']+h/2];pos[axis]+=sign*([sx,sy][axis]/2+t/2);dims=[sx+2*t,sy+2*t,h];dims[axis]=t;w[f'wall_{axis}_{sign}']=dict(dims=dims,pose=[*pos,1,0,0,0])
    return w

def robot_model():
    robot=load_yaml(str(BASE/'task_000759/sg2_right.yml'))['robot_cfg']
    robot['kinematics']['collision_spheres']['tcp']=[dict(center=[0,s,z],radius=.016) for s in [-.063,.063] for z in [.025,.055,.085]]+[dict(center=[0,0,.12],radius=.035)]
    return robot

def fit(q,dt,reference,profile,stage,speed=1.):
    if profile=='reference':weight=.25;travel=0.
    elif profile=='natural':weight=0.;travel=.001
    else:
        weight=(lambda u:.005+.075*np.clip((u-.75)/.25,0,1)**2) if stage=='reach' else .005
        travel=.0005
    result=prior_spline(q,dt,reference,weight,travel_regularization=travel)
    import os
    if os.environ.get('TASK759_TIMING_PROFILE')=='rapid':
        from rapid_timing import retime
        q,v,a,info=result;q,v,a,timing=retime(q,v,a,stage,float(os.environ.get('TASK759_RAPID_SLOWDOWN','1')))
        info.update(timing);return q,v,a,info
    if speed==1.:return result
    from scipy.interpolate import BPoly
    q,v,a,info=result
    old_time=np.arange(len(q))/60
    duration=np.ceil(old_time[-1]/speed*60)/60
    scale=old_time[-1]/duration
    polynomial=BPoly.from_derivatives(old_time,[np.array([p,d,dd]) for p,d,dd in zip(q,v,a)])
    grid=np.linspace(0,old_time[-1],int(round(duration*60))+1)
    info.update(unaccelerated_duration_s=info['duration_s'],duration_s=duration,requested_motion_speed=speed,effective_motion_speed=scale)
    for order,key in [(1,'analytic_peak_velocity_rad_s'),(2,'analytic_peak_acceleration_rad_s2'),(3,'analytic_peak_jerk_rad_s3')]:info[key]*=scale**order
    if info['analytic_peak_velocity_rad_s']>1.5 or info['analytic_peak_acceleration_rad_s2']>4 or info['analytic_peak_jerk_rad_s3']>40:raise ValueError('Retiming exceeds configured planning bounds')
    return polynomial(grid),polynomial(grid,1)*scale,polynomial(grid,2)*scale**2,info
