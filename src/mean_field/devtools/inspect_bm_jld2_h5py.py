from __future__ import annotations

import argparse

import h5py


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect whether a JLD2 file is readable through h5py.")
    parser.add_argument("path")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as handle:
        print(f"path={args.path}")
        print("top_level_keys=" + ",".join(sorted(handle.keys())))
        for key in sorted(handle.keys()):
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                print(f"{key}\tdataset\tshape={obj.shape}\tdtype={obj.dtype}")
            else:
                print(f"{key}\tgroup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
