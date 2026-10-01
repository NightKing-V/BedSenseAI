import pandas as pd
import numpy as np

df = pd.read_csv('outputs/sample4/telemetry.csv')

def print_detailed(st, et, label):
    sub = df[(df['timestamp_sec'] >= st) & (df['timestamp_sec'] <= et)]
    print(f"\n--- {label} ({st}s - {et}s) ---")
    for _, r in sub.iloc[::8].iterrows():
        sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
        sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
        hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
        hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
        
        dx = hp_x - sh_x
        dy = hp_y - sh_y
        inc = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else np.nan
        w = r['person_bbox_w']
        h = r['person_bbox_h']
        aspect = h / max(w, 1e-3)
        print(f"t={r['timestamp_sec']:5.2f}s | state={r['instantaneous_state']:16s} | bbox=({w:4.0f}x{h:4.0f}, asp={aspect:3.1f}) | hip={r['hip_angle_deg']} | knee={r['knee_angle_deg']} | inc={inc:4.1f}° | aff={r.get('mean_bed_affinity', 0.0):4.2f} | vel={r['body_center_velocity']:.4f}")

print_detailed(20.0, 30.0, "Lying in Bed (Sample 4)")
print_detailed(52.0, 58.0, "Sitting on Bed (Sample 4)")
print_detailed(130.0, 140.0, "Lying in Bed 2 (Sample 4)")
print_detailed(193.0, 199.0, "Sitting on Bed 2 (Sample 4)")
