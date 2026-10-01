import pandas as pd
import json
import numpy as np

df = pd.read_csv('outputs/sample5/telemetry.csv')

# Look at bed bounding box
bed_x1 = df['bed_bbox_x1'].iloc[0]
bed_y1 = df['bed_bbox_y1'].iloc[0]
bed_x2 = df['bed_bbox_x2'].iloc[0]
bed_y2 = df['bed_bbox_y2'].iloc[0]
print(f"Bed BBox: ({bed_x1:.1f}, {bed_y1:.1f}, {bed_x2:.1f}, {bed_y2:.1f}) | W={bed_x2-bed_x1:.1f}, H={bed_y2-bed_y1:.1f}")

sub = df[(df['timestamp_sec'] >= 11.5) & (df['timestamp_sec'] <= 20.0)]
print("\nTimestamp | State        | BBox(W x H) | Aspect(H/W) | Hip Ang | Knee Ang | Torso Norm | Torso Angle (deg) | Head-Torso Y-span")
print("-" * 115)

for _, r in sub.iloc[::2].iterrows():
    w, h = r['person_bbox_w'], r['person_bbox_h']
    aspect = h / max(w, 1e-3)
    sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
    sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
    hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
    hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
    
    dx = hp_x - sh_x
    dy = hp_y - sh_y
    # Torso vector angle from horizontal: dy is downward, dx is rightward
    # If torso is vertical (head/shoulders above hips), dy > 0 and dx ~ 0 => angle ~ 90 deg!
    torso_angle = np.degrees(np.arctan2(dy, dx)) if pd.notna(dx) else np.nan
    
    # Let's also check vertical torso inclination from horizontal plane:
    # abs(dy) / sqrt(dx^2 + dy^2) or angle from horizontal plane
    inclination_deg = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else np.nan

    hip_str = f"{r['hip_angle_deg']:.1f}°" if pd.notna(r['hip_angle_deg']) and str(r['hip_angle_deg']) != '' else "N/A"
    knee_str = f"{r['knee_angle_deg']:.1f}°" if pd.notna(r['knee_angle_deg']) and str(r['knee_angle_deg']) != '' else "N/A"
    t_norm = f"{r['torso_length_normalized']:.2f}" if pd.notna(r['torso_length_normalized']) else "N/A"
    
    print(f"{r['timestamp_sec']:5.2f}s    | {r['instantaneous_state']:12s} | {w:5.1f}x{h:5.1f}   | {aspect:5.2f}       | {hip_str:7s} | {knee_str:8s} | {t_norm:10s} | {torso_angle:6.1f}° (inc:{inclination_deg:4.1f}°) | sh_y:{sh_y:.0f} hip_y:{hp_y:.0f}")
