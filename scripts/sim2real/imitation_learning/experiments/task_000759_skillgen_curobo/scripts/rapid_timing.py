"""Simulation-only per-phase dynamic retiming; keep 60 Hz, do not drop frames.

These experimental acceleration/jerk caps are higher than the earlier conservative
planning settings. They are not verified hardware limits or a sim-to-real claim.
"""
import numpy as np
from scipy.interpolate import BPoly

LIMITS={'reach':(1.8,7.5,110.),'approach':(.45,2.,30.),'lift':(1.4,7.,100.),'close_lift':(.8,5.,80.),'carry':(1.8,7.5,110.),'descent':(1.2,5.,80.),'withdraw':(1.8,7.5,110.),'home':(1.8,7.5,110.)}

def retime(q,v,a,stage='reach',slowdown=1.):
    q=np.asarray(q,dtype=float);v=np.asarray(v,dtype=float);a=np.asarray(a,dtype=float)
    t=np.arange(len(q))/60;poly=BPoly.from_derivatives(t,[np.array([p,d,dd]) for p,d,dd in zip(q,v,a)])
    grid=np.linspace(0,t[-1],max(2000,len(q)*5))
    peaks=np.array([np.abs(poly(grid,k)).max() for k in [1,2,3]])
    limits=np.array(LIMITS.get(stage,LIMITS['carry']))
    dilation=max(peaks[0]/limits[0],np.sqrt(peaks[1]/limits[1]),np.cbrt(peaks[2]/limits[2]))*1.03*slowdown
    duration=np.ceil(max(.20,t[-1]*dilation)*60)/60
    ratio=t[-1]/duration;out_grid=np.linspace(0,t[-1],int(round(duration*60))+1)
    info=dict(duration_s=duration,input_duration_s=float(t[-1]),time_scale=float(ratio),stage=stage,limits=dict(velocity_rad_s=float(limits[0]),acceleration_rad_s2=float(limits[1]),jerk_rad_s3=float(limits[2])),peak_velocity_rad_s=float(peaks[0]*ratio),peak_acceleration_rad_s2=float(peaks[1]*ratio**2),peak_jerk_rad_s3=float(peaks[2]*ratio**3),experimental_simulation_only=True)
    return poly(out_grid),poly(out_grid,1)*ratio,poly(out_grid,2)*ratio**2,info

def monotone(q0,q1,stage='close_lift'):
    t=np.linspace(0,.5,31);u=t/.5;s=10*u**3-15*u**4+6*u**5
    ds=(30*u**2-60*u**3+30*u**4)/.5;dds=(60*u-180*u**2+120*u**3)/.5**2
    delta=np.asarray(q1)-q0
    return retime(q0+s[:,None]*delta,ds[:,None]*delta,dds[:,None]*delta,stage)
