"""Join all eleven accepted final videos, label top right, explicit playback speed.

Preserves every source video frame: 15 fps source -> 45 fps export. No generated
intermediate frames, cut task phases, frozen ending, or source file modification.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2,imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--folder',default='/experiment/output/hybrid11_report');p.add_argument('--speed',type=float,default=3.);args=p.parse_args()
folder=Path(args.folder);summary=json.loads((folder/'summary.json').read_text())
assert summary['successful']==11 and len(summary['runs'])==11
stem=f'final_11_angles_{args.speed:g}x'
output=folder/f'{stem}.mp4';font=ImageFont.load_default(size=30)
timeline=[];frames_written=0;source_fps=15.;output_fps=source_fps*args.speed
with imageio.get_writer(output,fps=output_fps,codec='libx264',quality=None,pixelformat='yuv420p',macro_block_size=16,ffmpeg_params=['-crf','18','-preset','medium','-movflags','+faststart']) as writer:
    for run in summary['runs']:
        assert run['success'] and run['return_home_success']
        source=Path(run['exported_video']);cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS)
        assert abs(fps-source_fps)<1e-6
        start=frames_written;count=0;yaw=run['yaw_deg'];label=f'{yaw:+d}\N{DEGREE SIGN}' if yaw else '0\N{DEGREE SIGN}'
        while True:
            ok,frame=cap.read()
            if not ok:break
            rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));assert rgb.size==(640,480)
            draw=ImageDraw.Draw(rgb);box=draw.textbbox((0,0),label,font=font);tw=box[2]-box[0];th=box[3]-box[1]
            x=640-16-tw-16;y=14
            draw.rounded_rectangle((x,y,624,y+th+24),radius=8,fill=(13,19,27))
            draw.text((x+8,y+12-box[1]),label,font=font,fill=(255,255,255))
            writer.append_data(np.asarray(rgb));frames_written+=1;count+=1
        cap.release();assert count==run['video_frames']
        timeline.append(dict(yaw_deg=yaw,source=str(source),source_frames=count,output_start_frame=start,output_end_frame_exclusive=frames_written,output_start_s=start/output_fps,output_end_s=frames_written/output_fps,full_pick_place_return=True))
        print('JOINED',yaw,count,flush=True)
expected=sum(r['video_frames'] for r in summary['runs']);assert frames_written==expected
cap=cv2.VideoCapture(str(output));actual_fps=cap.get(cv2.CAP_PROP_FPS);decoded=0
snapshots={};selected=set()
for segment in timeline:selected.update([segment['output_start_frame'],segment['output_end_frame_exclusive']-1])
while True:
    ok,frame=cap.read()
    if not ok:break
    if decoded in selected:snapshots[decoded]=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    decoded+=1
cap.release();assert decoded==expected and abs(actual_fps-output_fps)<1e-6
sheet=Image.new('RGB',(1280,240*6),(15,20,27));draw=ImageDraw.Draw(sheet)
for index,segment in enumerate(timeline):
    # First frame of each labeled segment; final return frame for the last.
    image=snapshots[segment['output_start_frame']].resize((640,480)).resize((426,320))
    x=(index%3)*426;y=(index//3)*320;sheet.paste(image,(x,y))
sheet.save(folder/f'{stem}_review.jpg')
report=dict(video=str(output),speed=args.speed,source_fps=source_fps,output_fps=actual_fps,source_total_duration_s=expected/source_fps,duration_s=decoded/actual_fps,frames=decoded,all_frames_decoded=True,all_source_frames_preserved=True,no_phase_trimmed=True,top_right_angle_label=True,angle_order=[s['yaw_deg'] for s in timeline],timeline=timeline,sha256=hashlib.sha256(output.read_bytes()).hexdigest())
(folder/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
