#!/data/home/ziyuzhu/miniconda3/bin/python -S
"""Publish one non-authoritative terminal failure record."""
from __future__ import annotations
import json, os, sys
from pathlib import Path


def main():
    if len(sys.argv)!=4: return 2
    root=Path(sys.argv[1]).resolve(strict=True)
    job_id=os.environ.get("SLURM_JOB_ID","UNKNOWN")
    path=root/"output"/f"DIAGNOSTIC_INCOMPLETE_{job_id}.json"
    payload={
        "schema":"mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_incomplete.v24",
        "status":"diagnostic_incomplete",
        "job_id":job_id,
        "failure_stage":sys.argv[2],
        "exit_code":int(sys.argv[3]),
        "scientific_authority_promoted":False,
        "positive_verdict_permitted":False,
        "result_reuse_permitted":False,
    }
    data=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0),0o400)
    except FileExistsError: return 0
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0: raise OSError("short failure publication write")
            view = view[written:]
        os.fsync(fd)
    finally: os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    return 0

if __name__=="__main__": raise SystemExit(main())
