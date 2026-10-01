import pandas as pd
import numpy as np

def evaluate_classifier_on_df(csv_path, label):
    df = pd.read_csv(csv_path)
    
    # Run evidence logic frame by frame
    states = []
    
    for _, r in df.iterrows():
        # Check if missing
        if pd.isna(r['person_bbox_w']) or r['person_bbox_w'] <= 0:
            states.append('UNKNOWN')
            continue
            
        sh_x = (r['left_shoulder_x'] + r['right_shoulder_x']) / 2 if pd.notna(r['left_shoulder_x']) else np.nan
        sh_y = (r['left_shoulder_y'] + r['right_shoulder_y']) / 2 if pd.notna(r['left_shoulder_y']) else np.nan
        hp_x = (r['left_hip_x'] + r['right_hip_x']) / 2 if pd.notna(r['left_hip_x']) else np.nan
        hp_y = (r['left_hip_y'] + r['right_hip_y']) / 2 if pd.notna(r['left_hip_y']) else np.nan
        
        dx = hp_x - sh_x
        dy = hp_y - sh_y
        inc = np.degrees(np.arctan2(abs(dy), abs(dx))) if pd.notna(dx) and (dx**2 + dy**2) > 0 else None
        
        hip_angle = r['hip_angle_deg'] if pd.notna(r['hip_angle_deg']) and r['hip_angle_deg'] != '' else None
        knee_angle = r['knee_angle_deg'] if pd.notna(r['knee_angle_deg']) and r['knee_angle_deg'] != '' else None
        
        body_vel = r['body_center_velocity'] if pd.notna(r['body_center_velocity']) else 0.0
        ankle_vel = r['ankle_velocity'] if pd.notna(r['ankle_velocity']) else 0.0
        knee_vel = r['knee_velocity'] if pd.notna(r['knee_velocity']) else 0.0
        
        aff = r.get('mean_bed_affinity', 0.0)
        if pd.isna(aff): aff = 0.0
        bed_overlap = r['bed_overlap'] if pd.notna(r['bed_overlap']) else 0.0
        
        on_bed = (aff >= 0.25) or (bed_overlap >= 0.30)
        off_bed = (aff <= 0.12) and (bed_overlap < 0.20)
        
        is_horizontal = (inc is not None and inc <= 45.0)
        
        bent_hip_angle = 135.0
        extended_hip_angle = 155.0
        bent_knee_angle = 135.0
        extended_knee_angle = 150.0
        low_body_motion = 0.020
        high_body_motion = 0.035
        
        has_bent_hip = (hip_angle is not None and hip_angle < bent_hip_angle)
        has_extended_hip = (hip_angle is not None and hip_angle >= extended_hip_angle)
        has_bent_knee = (knee_angle is not None and knee_angle < bent_knee_angle)
        has_extended_knee = (knee_angle is not None and knee_angle >= extended_knee_angle)
        
        is_moving = (body_vel > high_body_motion) or (ankle_vel > high_body_motion) or (knee_vel > high_body_motion)
        is_resting = (body_vel < low_body_motion)
        
        lying = 0.0
        sitting = 0.0
        standing = 0.0
        walking = 0.0
        
        # 1. LYING EVIDENCE
        if on_bed:
            if is_horizontal:
                lying += 0.70
                if is_resting: lying += 0.20
                if has_extended_knee or has_extended_hip: lying += 0.10
            elif has_bent_hip:
                lying += 0.05
            elif is_moving and not off_bed:
                # Moving upright near/on bed (e.g. standing up from bed or walking)
                if has_extended_hip or has_extended_knee:
                    walking += 0.50
                    lying += 0.10
                else:
                    lying += 0.40
            else:
                # Longitudinal/angled in bed: extended body or resting posture
                lying += 0.60
                if has_extended_hip: lying += 0.20
                if has_extended_knee: lying += 0.10
                if is_resting: lying += 0.15
        elif is_horizontal:
            lying += 0.40
            if is_resting: lying += 0.30
            
        # 2. SITTING EVIDENCE
        if on_bed:
            if has_bent_hip:
                sitting += 0.70
                if has_bent_knee: sitting += 0.20
                if is_resting: sitting += 0.10
            elif has_bent_knee and not has_extended_hip:
                sitting += 0.45
                if is_resting: sitting += 0.15
            else:
                sitting += 0.05
        elif not off_bed:
            if has_bent_hip or has_bent_knee:
                sitting += 0.55
                if is_resting: sitting += 0.20
            elif not is_horizontal and is_resting:
                sitting += 0.20
        else:
            if has_bent_hip or has_bent_knee:
                sitting += 0.60
                if is_resting: sitting += 0.25
                
        # 3. STANDING EVIDENCE
        if not is_horizontal:
            if is_resting:
                if off_bed:
                    standing += 0.45
                    if has_extended_hip: standing += 0.25
                    if has_extended_knee: standing += 0.25
                elif not on_bed:
                    standing += 0.25
                    if has_extended_hip: standing += 0.25
                    
        # 4. WALKING EVIDENCE
        if not is_horizontal and is_moving:
            if off_bed:
                walking += 0.40
            elif not on_bed:
                walking += 0.25
            else:
                walking += 0.15
                
            if body_vel > high_body_motion: walking += 0.35
            if ankle_vel > high_body_motion: walking += 0.25
            if knee_vel > high_body_motion: walking += 0.15
            
        # Winner
        scores = {"LYING_IN_BED" if on_bed else "LYING_IN_BED": lying,
                  "SITTING_ON_BED" if on_bed else "SITTING_OUTSIDE_BED": sitting,
                  "STANDING": standing,
                  "WALKING": walking}
        winner = max(scores, key=scores.get)
        states.append(winner)
        
    df['simulated_state'] = states
    
    # Print state segments
    print(f"\n=================== {label} ===================")
    current_state = states[0]
    start_t = df['timestamp_sec'].iloc[0]
    for i in range(1, len(states)):
        if states[i] != current_state:
            end_t = df['timestamp_sec'].iloc[i-1]
            dur = end_t - start_t
            if dur >= 0.5:
                print(f"[{start_t:5.2f}s -> {end_t:5.2f}s] {current_state:18s} ({dur:4.1f}s)")
            current_state = states[i]
            start_t = df['timestamp_sec'].iloc[i]
    end_t = df['timestamp_sec'].iloc[-1]
    dur = end_t - start_t
    print(f"[{start_t:5.2f}s -> {end_t:5.2f}s] {current_state:18s} ({dur:4.1f}s)")

evaluate_classifier_on_df('outputs/sample5/telemetry.csv', "SAMPLE 5 SIMULATION")
evaluate_classifier_on_df('outputs/sample2/telemetry.csv', "SAMPLE 2 SIMULATION")
evaluate_classifier_on_df('outputs/sample4/telemetry.csv', "SAMPLE 4 SIMULATION")
