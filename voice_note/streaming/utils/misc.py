from pathlib import Path
from datetime import datetime


def round_to_nearest_appropriate_number(num, step_size):
    """ Rounds the given number `num` to the nearest multiple of `step_size` """
    return round(num / step_size) * step_size


def prepare_log_file(file_path):
    if file_path is None:
        return

    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
    file_path.parent.mkdir(parents=True, exist_ok=True)


def log_bytes(bytes_, file_handle):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    hex = ' '.join(f"{b:02x}" for b in bytes_)
    file_handle.write(f"{timestamp}: {hex}\n")
