"""Remove only verified macOS AppleDouble sidecars inside a virtual environment.

On external volumes, ._*.dist-info and ._*.mplstyle may be mistaken for package
metadata or matplotlib styles. Actual package files are always preserved.
"""
from argparse import ArgumentParser
from pathlib import Path


def clean_venv(path: Path) -> int:
    if not (path / 'pyvenv.cfg').is_file():
        raise ValueError(f'Not a virtual environment: {path}')
    removed = 0
    for sidecar in path.rglob('._*'):
        if sidecar.is_symlink() or not sidecar.is_file():
            continue
        with sidecar.open('rb') as stream:
            magic = stream.read(4)
        if magic == b'\x00\x05\x16\x07':
            sidecar.unlink()
            removed += 1
    return removed


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('venv', nargs='?', type=Path, default=Path('.venv'))
    args = parser.parse_args()
    print(f'AppleDouble sidecars removed: {clean_venv(args.venv)}')
