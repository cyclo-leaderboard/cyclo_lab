"""Check that an MP4 preview was finalized before publishing or reusing it."""
from pathlib import Path


def is_complete_mp4(path: Path) -> bool:
    """Check top-level box boundaries and required metadata, without decoding."""
    try:
        size = path.stat().st_size
        boxes = set()
        position = 0
        with path.open('rb') as stream:
            while position < size:
                stream.seek(position)
                header = stream.read(8)
                if len(header) != 8:
                    return False
                length = int.from_bytes(header[:4], 'big')
                kind = header[4:]
                header_size = 8
                if length == 1:
                    extended = stream.read(8)
                    if len(extended) != 8:
                        return False
                    length = int.from_bytes(extended, 'big')
                    header_size = 16
                elif length == 0:
                    length = size - position
                if length < header_size or position + length > size:
                    return False
                if length > header_size:
                    boxes.add(kind)
                position += length
        return {b'ftyp', b'moov', b'mdat'}.issubset(boxes)
    except OSError:
        return False
