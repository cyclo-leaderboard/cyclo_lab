"""Small mathematical checks; these do not establish physical or natural motion."""
import json
import numpy as np
from conditional_common import fit
from plan_goalset import BASE

source=np.load(BASE/'input/source_episode.npz');q=source['state'][:41,8:15];reference=q.copy();reports=[]
for profile in ['reference','natural','hybrid']:
    fitted,v,a,r=fit(q,1/15,reference,profile,'reach')
    assert np.isfinite(fitted).all()
    assert np.max(np.abs(fitted[0]-q[0]))<1e-8
    assert np.max(np.abs(fitted[-1]-q[-1]))<1e-8
    assert np.abs(v[[0,-1]]).max()<1e-8
    assert np.abs(a[[0,-1]]).max()<1e-8
    assert r['analytic_peak_velocity_rad_s']<=.8+1e-6
    assert r['analytic_peak_acceleration_rad_s2']<=2.+1e-6
    assert r['analytic_peak_jerk_rad_s3']<=15.+1e-6
    reports.append(dict(profile=profile,**r))
(BASE/'output/conditional_fit_unit_checks.json').write_text(json.dumps(reports,indent=2));print('3 PROFILES: FINITE, ENDPOINTS, C2, COMMAND DERIVATIVE LIMITS PASS')
