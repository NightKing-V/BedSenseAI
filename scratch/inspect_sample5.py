import pandas as pd
import numpy as np

df = pd.read_csv('outputs/sample5/telemetry.csv')
print("Columns:", df.columns.tolist())

print("\n=== INSPECTING t = 18.0 to 24.0 ===")
sub1 = df[(df['timestamp_sec'] >= 18.0) & (df['timestamp_sec'] <= 24.0)]
for _, r in sub1.iterrows():
    sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
    sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
    hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
    hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
    dx = hp_x - sh_x
    dy = hp_y - sh_y
    inc = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else np.nan

    print(f"t={r['timestamp_sec']:5.2f}s | state={r['instantaneous_state']:18s} | bed_overlap={r['bed_overlap']:4.2f} | aff={r.get('mean_bed_affinity', 0.0):4.2f} | vel={r['body_center_velocity']:.4f} | ank_vel={r['ankle_velocity']:.4f} | inc={inc:4.1f}° | hip={r['hip_angle_deg']}")

print("\n=== INSPECTING t = 30.0 to 39.0 ===")
sub2 = df[(df['timestamp_sec'] >= 30.0) & (df['timestamp_sec'] <= 39.0)]
for _, r in sub2.iterrows():
    sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
    sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
    hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
    hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
    dx = hp_x - sh_x
    dy = hp_y - sh_y
    inc = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else np.nan
    print(f"t={r['timestamp_sec']:5.2f}s | state={r['instantaneous_state']:18s} | bed_overlap={r['bed_overlap']:4.2f} | aff={r.get('mean_bed_affinity', 0.0):4.2f} | hip_ang={r['hip_angle_deg']} | knee_ang={r['knee_angle_deg']} | inc={inc:4.1f}° | vel={r['body_center_velocity']:.4f}")
