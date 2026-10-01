import pandas as pd
import json

df = pd.read_csv('outputs/sample4/telemetry.csv')
with open('data/sample4.json') as f:
    gt = json.load(f)

for seg in gt:
    st, et, state = seg['start_t'], seg['end_t'], seg['state']
    sub = df[(df['timestamp_sec'] >= st) & (df['timestamp_sec'] <= et)]
    if len(sub) == 0:
        continue
    mean_hip = sub['hip_angle_deg'].dropna().mean()
    mean_knee = sub['knee_angle_deg'].dropna().mean()
    mean_vel = sub['body_center_velocity'].dropna().mean()
    mean_aff = sub['mean_bed_affinity'].dropna().mean() if 'mean_bed_affinity' in sub else 0
    print(f"[{st:5.1f}s -> {et:5.1f}s] GT: {state:18s} | count={len(sub):3d} | hip={mean_hip:5.1f}° | knee={mean_knee:5.1f}° | vel={mean_vel:.4f} | aff={mean_aff:.2f}")
