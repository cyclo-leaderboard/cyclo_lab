"""Deterministic random review and extraction of exactly one LeRobot v3 episode."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('--episode', type=int)
a = p.parse_args()
root, out = Path('/dataset'), Path('/experiment/input')
info = json.loads((root/'meta/info.json').read_text())
episodes = pq.read_table(root/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
order = np.random.default_rng(759).permutation(len(episodes)).tolist()
def video_path(row, camera):
    key = 'observation.images.rgb.' + camera
    return root/info['video_path'].format(video_key=key, chunk_index=row[f'videos/{key}/chunk_index'], file_index=row[f'videos/{key}/file_index'])
def get_frame(row, camera, local_frame):
    path = video_path(row, camera)
    cap = cv2.VideoCapture(str(path))
    start = round(row[f'videos/observation.images.rgb.{camera}/from_timestamp'] * info['fps'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start+local_frame)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f'Cannot decode {path}, frame {local_frame}')
    return frame
if a.episode is None:
    rows = []
    for j in range(0, 24, 4):
        cells = []
        for idx in order[j:j+4]:
            row = episodes[idx]
            frame = cv2.resize(get_frame(row, 'cam_left_head', 0), (384, 216))
            cv2.putText(frame, f'random rank {order.index(idx)} / episode {idx}', (8,24), 0, .65, (0,0,255),2)
            cells.append(frame)
        rows.append(np.concatenate(cells,axis=1))
    cv2.imwrite(str(out/'random_candidates.jpg'), np.concatenate(rows,axis=0))
    (out/'random_order.json').write_text(json.dumps({'seed':759,'order':order},indent=2))
    print('RANDOM_ORDER', order[:24])
else:
    row = next(r for r in episodes if r['episode_index']==a.episode)
    path = root/info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])
    table = pq.read_table(path)
    selected = table.filter(pc.equal(table['episode_index'], a.episode))
    assert len(selected)==row['length']
    q = np.asarray(selected['observation.state'].to_pylist(),dtype=np.float32)
    u = np.asarray(selected['action'].to_pylist(),dtype=np.float32)
    t = np.asarray(selected['timestamp'].to_pylist(),dtype=float)
    np.savez(out/'source_episode.npz', state=q,action=u,time_sec=t-t[0],joint_names=np.array(info['features']['observation.state']['names']),global_index=np.asarray(selected['index']))
    pq.write_table(selected,out/'source_episode.parquet')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    provenance = {'dataset_host':'/home/robotis-ai/cyclo_intelligence/docker/workspace/dataset/Dongkkka/task_000759_move_vegetable_v30', 'selection_seed':759,'random_rank':order.index(a.episode),'episode_index':a.episode,'source_episode_index':row['source_episode_index'],'frames':len(q),'fps':info['fps'],'metadata':{k:v for k,v in row.items() if not k.startswith('stats/')},'source_parquet_sha256':sha(path),'action_max_abs_state_difference':float(np.max(abs(q-u))),'base_velocity_max_abs':np.max(abs(q[:,19:]),axis=0).tolist(),'source_outcome':'pending visual verification; metadata has no success flag'}
    cells=[]
    for cam in ['cam_left_head','cam_left_wrist','cam_right_wrist']:
        first=get_frame(row,cam,0)
        writer=imageio_ffmpeg.write_frames(str(out/f'source_{cam}.mp4'),(first.shape[1],first.shape[0]),fps=15,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,quality=7)
        writer.send(None)
        cap=cv2.VideoCapture(str(video_path(row,cam)))
        start=round(row[f'videos/observation.images.rgb.{cam}/from_timestamp']*15)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        for i in range(len(q)):
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Missing frame {i}')
            writer.send(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release();writer.close()
    for idx in np.linspace(0,len(q)-1,12,dtype=int):
        frame=cv2.resize(get_frame(row,'cam_left_head',int(idx)),(384,216))
        cv2.putText(frame,f'episode {a.episode} frame {idx}',(8,24),0,.65,(0,0,255),2)
        cells.append(frame)
    cv2.imwrite(str(out/'source_review.jpg'),np.concatenate([np.concatenate(cells[i:i+4],axis=1) for i in range(0,12,4)],axis=0))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in provenance.items() if k!='metadata'},indent=2))
