"""Decode every saved MP4 frame; produce a contact sheet for human visual review."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('video');a=p.parse_args();path=Path(a.video)
cap=cv2.VideoCapture(str(path));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);indices=set(np.linspace(0,expected-1,min(12,expected)).astype(int));frames=[];count=0
while True:
    ok,bgr=cap.read()
    if not ok:break
    if count in indices:frames.append((count,Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((480,360))))
    count+=1
cap.release();assert count==expected and count>0,(count,expected)
sheet=Image.new('RGB',(1920,360*((len(frames)+3)//4)))
for k,(index,img) in enumerate(frames):
    x=(k%4)*480;y=(k//4)*360;sheet.paste(img,(x,y));ImageDraw.Draw(sheet).text((x+8,y+8),f'frame {index}',fill='red')
sheet.save(path.with_name(path.stem+'_review.jpg'))
report={'video':str(path),'decoded_frames':count,'fps':fps,'duration_s':count/fps,'all_frames_decoded':True}
path.with_name(path.stem+'_decode.json').write_text(json.dumps(report,indent=2));print(report)
