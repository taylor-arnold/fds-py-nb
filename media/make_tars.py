import tarfile
from pathlib import Path

media_dir = Path(__file__).parent

for d in sorted(media_dir.iterdir()):
    if d.is_dir():
        tar_path = media_dir / f"{d.name}.tar"
        print(f"Creating {tar_path.name} ...")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(d, arcname=d.name)
        print(f"  done ({tar_path.stat().st_size / 1e6:.1f} MB)")
