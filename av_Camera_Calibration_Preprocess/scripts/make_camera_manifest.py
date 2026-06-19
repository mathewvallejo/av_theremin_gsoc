import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a camera manifest CSV from calibration .npz files."
    )
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.calibration_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        rows.append({
            "camera_id": str(data["camera_id"]),
            "calibration_path": str(path),
            "image_width": int(data["image_width"]),
            "image_height": int(data["image_height"]),
            "reprojection_error": float(data["reprojection_error"]),
            "usable_images": int(data["usable_images"]),
            "total_images": int(data["total_images"]),
        })

    if not rows:
        raise RuntimeError(f"No .npz calibration files found in {args.calibration_dir}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"Cameras in manifest: {len(rows)}")


if __name__ == "__main__":
    main()
