"""Latest-frame CPU encoding for the live UI (no simulator calls in the worker)."""

import os
import threading


class LiveCameraPreviews:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.last_frames = {}
        self._condition = threading.Condition()
        self._pending = {}
        self._published = set()
        self._sequence = 0
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="preview-jpeg", daemon=True)
        self._thread.start()

    def submit(self, images):
        with self._condition:
            # One pending image per camera; a newer capture replaces an old one.
            self._pending.update(images)
            self._condition.notify()

    def snapshot(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"Live preview encoding failed: {self._error}") from self._error
            return self._sequence, sorted(self._published)

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify()
        self._thread.join()

    def _run(self):
        try:
            import cv2
            import numpy as np

            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    batch, self._pending = self._pending, {}
                for label, (image, turns, limit) in batch.items():
                    if turns:
                        image = np.rot90(image, turns)
                    if image.dtype != np.uint8:
                        if image.size and float(image.max()) <= 1.0:
                            image = image * 255.0
                        image = image.clip(0, 255).astype(np.uint8)
                    if limit is not None:
                        scale = min(1.0, limit[0] / image.shape[1], limit[1] / image.shape[0])
                        if scale < 1.0:
                            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                                       max(1, round(image.shape[0] * scale))),
                                               interpolation=cv2.INTER_AREA)
                    options = [cv2.IMWRITE_JPEG_QUALITY, 70] if limit is not None else []
                    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), options)
                    if not ok:
                        raise RuntimeError(f"Could not encode {label}")
                    destination = self.directory / f"{label}.jpg"
                    temporary = destination.with_suffix(".jpg.tmp")
                    temporary.write_bytes(buffer.tobytes())
                    os.replace(temporary, destination)
                # Announce a completed batch, not a submitted simulator step.
                with self._condition:
                    self._published.update(batch)
                    self._sequence += 1
        except Exception as exc:
            with self._condition:
                self._error = exc
