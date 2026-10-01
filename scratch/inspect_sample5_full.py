import pandas as pd
import numpy as np

df = pd.read_csv('outputs/sample5/telemetry.csv')

def analyze_range(start_t, end_t, label):
    sub = df[(df['timestamp_sec'] >= start_t) & (df['timestamp_sec'] <= end_t)]
    print(f"\n=================== {label} ({start_t}s - {end_t}s) ===================")
    for _, r in sub.iloc[::5].iterrows():
        sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
        sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
        hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
        hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
        kn_y = (r['left_knee_y'] + r['right_knee_y']) / 2 if pd.notna(r['left_knee_y']) else np.nan
        ak_y = (r['left_ankle_y'] + r['right_ankle_y']) / 2 if pd.notna(r['left_ankle_y']) else np.nan
        
        dx = hp_x - sh_x
        dy = hp_y - sh_y
        inc = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else np.nan
        
        w = r['person_bbox_w']
        h = r['person_bbox_h']
        sh_w = r['shoulder_width'] if 'shoulder_width' in r and pd.notna(r['shoulder_width']) else np.nan
        
        print(f"t={r['timestamp_sec']:5.2f}s | state={r['instantaneous_state']:16s} | bbox=({w:4.0f}x{h:4.0f}) | hip_ang={r['hip_angle_deg']:5.1f} | knee_ang={r['knee_angle_deg']:5.1f} | inc={inc:4.1f}° | sh_y={sh_y:4.0f} hp_y={hp_y:4.0f} kn_y={kn_y:4.0f} ak_y={ak_y:4.0f} | vel={r['body_center_velocity']:.4f}")

analyze_range(0.0, 11.0, "1. Lying in bed (initial)")
analyze_range(12.0, 19.0, "2. Sitting on bed (sitting upright)")
analyze_range(19.5, 21.5, "3. Walking (away)")
analyze_range(25.5, 27.5, "4. Walking & Sitting on bed")
analyze_range(28.0, 33.0, "5. Lying in bed (turn 1)")
analyze_range(33.0, 38.0, "6. Lying in bed (turn 2 / full stretch)")
analyze_range(38.0, 39.0, "7. Lying in bed (final)")
