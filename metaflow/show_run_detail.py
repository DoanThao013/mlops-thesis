"""Hiển thị chi tiết 1 run cụ thể (giống Run detail trong UI)"""
import os
import sys

os.environ['METAFLOW_DEFAULT_DATASTORE'] = 'local'
os.environ['METAFLOW_DEFAULT_METADATA'] = 'local'

from metaflow import Run

if len(sys.argv) < 2:
    print("Usage: python3 show_run_detail.py <flow_name>/<run_id>")
    print("Example: python3 show_run_detail.py MNISTFlow/1779208836058684")
    sys.exit(1)

run = Run(sys.argv[1])
print(f"=== Run: {run.id} ===")
print(f"Successful: {run.successful}")
print(f"Created: {run.created_at}")
print(f"Finished: {run.finished_at}")

print(f"\n--- Steps ---")
for step in run.steps():
    print(f"  {step.id}: {len(list(step.tasks()))} task(s)")

print(f"\n--- Artifacts ---")
for key in sorted(run.data._artifacts.keys()):
    val = getattr(run.data, key)
    if isinstance(val, (int, float, str, bool)):
        print(f"  {key}: {val}")
    else:
        print(f"  {key}: <{type(val).__name__}>")
