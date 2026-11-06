"""Make a compact first/mid/final visual audit sheet from Task901 videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    videos = sorted(args.input.glob("success_demo_*_head_wrist.mp4"))
    if not videos:
        raise FileNotFoundError(f"No success videos under {args.input}")
    sample_frames = (0, 170, 470, 790, 980)
    cell_w, cell_h = 256, 293
    rows = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        cells = []
        for frame_index in sample_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_index} from {video}")
            frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"{video.stem.split('_')[2]} f{frame_index}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(frame)
        capture.release()
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Failed to write {args.output}")
    print(f"TASK901_CONTACT_SHEET rows={len(rows)} samples={len(sample_frames)} {args.output}")


if __name__ == "__main__":
    main()
