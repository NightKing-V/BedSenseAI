import os
import cv2
import time
import base64
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from IPython.display import display, HTML
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Calibrated Environment & Metric Homography
# ---------------------------------------------------------
@dataclass
class CalibratedEnvironment:
    bed_quad_img: np.ndarray
    H_img_to_metric: np.ndarray
    bed_width_cm: float = 100.0
    bed_length_cm: float = 200.0

    def project_point(self, pt: Tuple[float, float]) -> np.ndarray:
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_img_to_metric)[0][0]
        return dst

    def compute_bed_overlap(self, bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = map(int, bbox)
        pw = max(1, x2 - x1)
        ph = max(1, y2 - y1)
        person_area = float(pw * ph)

        rect_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        bed_poly = self.bed_quad_img.astype(np.int32)

        min_x = min(x1, int(np.min(bed_poly[:, 0])))
        min_y = min(y1, int(np.min(bed_poly[:, 1])))
        max_x = max(x2, int(np.max(bed_poly[:, 0])))
        max_y = max(y2, int(np.max(bed_poly[:, 1])))

        w = max(1, max_x - min_x + 1)
        h = max(1, max_y - min_y + 1)

        m_person = np.zeros((h, w), dtype=np.uint8)
        m_bed = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(m_person, [rect_poly - np.array([min_x, min_y])], 1)
        cv2.fillPoly(m_bed, [bed_poly - np.array([min_x, min_y])], 1)

        intersection = np.sum((m_person == 1) & (m_bed == 1))
        return float(intersection / person_area)

def auto_calibrate_scene(frame: np.ndarray, yolo_model: YOLO) -> CalibratedEnvironment:
    det = yolo_model(frame, classes=[59], conf=0.15, verbose=False)[0]
    
    if len(det.boxes) > 0:
        boxes = det.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[np.argmax(areas)]
    else:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w * 0.25, h * 0.20, w * 0.75, h * 0.85

    bed_quad = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float32)

    metric_dst = np.array([
        [0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(bed_quad, metric_dst)
    return CalibratedEnvironment(bed_quad_img=bed_quad, H_img_to_metric=H)

# ---------------------------------------------------------
# 2. Resident Tracker & Classifier
# ---------------------------------------------------------
class ResidentTracker:
    def __init__(self, max_lost_sec: float = 3.0):
        self.max_lost_sec = max_lost_sec
        self.primary_track_id: Optional[int] = None
        self.last_known_bbox: Optional[np.ndarray] = None
        self.last_seen_ts: float = 0.0
        self.is_initialized: bool = False

    def update(self, track_ids, boxes, kpts_batch, timestamp_sec, env):
        if len(track_ids) == 0:
            return None

        if not self.is_initialized or self.primary_track_id is None:
            overlaps = [env.compute_bed_overlap(b) for b in boxes]
            best_idx = int(np.argmax(overlaps))
            self.primary_track_id = int(track_ids[best_idx])
            self.last_known_bbox = boxes[best_idx]
            self.last_seen_ts = timestamp_sec
            self.is_initialized = True
            return self.primary_track_id, boxes[best_idx], kpts_batch[best_idx]

        if self.primary_track_id in track_ids:
            idx = int(np.where(track_ids == self.primary_track_id)[0][0])
            self.last_known_bbox = boxes[idx]
            self.last_seen_ts = timestamp_sec
            return self.primary_track_id, boxes[idx], kpts_batch[idx]

        if self.last_known_bbox is not None:
            c_prev = np.array([(self.last_known_bbox[0] + self.last_known_bbox[2])/2,
                               (self.last_known_bbox[1] + self.last_known_bbox[3])/2])
            centers = np.array([[(b[0] + b[2])/2, (b[1] + b[3])/2] for b in boxes])
            dists = np.linalg.norm(centers - c_prev, axis=1)
            closest_idx = int(np.argmin(dists))

            if dists[closest_idx] < 250.0:
                self.primary_track_id = int(track_ids[closest_idx])
                self.last_known_bbox = boxes[closest_idx]
                self.last_seen_ts = timestamp_sec
                return self.primary_track_id, boxes[closest_idx], kpts_batch[closest_idx]

        if len(track_ids) == 1:
            self.primary_track_id = int(track_ids[0])
            self.last_known_bbox = boxes[0]
            self.last_seen_ts = timestamp_sec
            return self.primary_track_id, boxes[0], kpts_batch[0]

        return None

class ActivityClassifier:
    def __init__(self, env: CalibratedEnvironment):
        self.env = env
        self.prev_metric_pos = None
        self.prev_ts = 0.0

    def evaluate(self, kpts, bbox, timestamp):
        if kpts.ndim == 3: kpts = kpts[0]

        mean_kpt_conf = float(np.mean(kpts[:, 2]))
        if mean_kpt_conf < 0.25:
            return {
                "state": "UNKNOWN", 
                "speed": 0.0, 
                "angle": 0.0, 
                "overlap": 0.0, 
                "trunk_len": 0.0, 
                "aspect_ratio": 0.0,
                "mean_kpt_conf": mean_kpt_conf  # <-- Added missing key
            }

        sh_mid = (kpts[5, :2] + kpts[6, :2]) / 2.0
        hip_mid = (kpts[11, :2] + kpts[12, :2]) / 2.0

        torso_vec = sh_mid - hip_mid
        torso_len_2d = np.linalg.norm(torso_vec)
        if torso_len_2d > 1e-4:
            unit_torso = torso_vec / torso_len_2d
            dot = np.clip(np.dot(unit_torso, np.array([0.0, -1.0])), -1.0, 1.0)
            torso_angle = float(np.degrees(np.arccos(abs(dot))))
        else:
            torso_angle = 0.0

        proj_sh = self.env.project_point(sh_mid)
        proj_hip = self.env.project_point(hip_mid)
        metric_trunk_len = float(np.linalg.norm(proj_sh - proj_hip))

        dt = max(1e-3, timestamp - self.prev_ts)
        metric_speed = 0.0
        if self.prev_metric_pos is not None:
            metric_speed = float(np.linalg.norm(proj_hip - self.prev_metric_pos) / dt)
        
        self.prev_metric_pos = proj_hip
        self.prev_ts = timestamp

        overlap = self.env.compute_bed_overlap(bbox)
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        aspect_ratio = bw / bh

        in_bed = overlap >= 0.35
        # Updated logic: Relaxed aspect ratio to > 0.55 to catch the tall/narrow bounding box when lying flat facing camera
        is_horizontal = (torso_angle >= 45.0) or (metric_trunk_len >= 40.0 and aspect_ratio > 0.55)
        is_walking = metric_speed >= 25.0

        if in_bed:
            state = "LYING_IN_BED" if is_horizontal else "SITTING_ON_BED"
        else:
            if is_walking:
                state = "WALKING"
            elif is_horizontal:
                state = "OUT_OF_BED"
            elif aspect_ratio < 0.60:
                state = "STANDING"
            else:
                state = "SITTING_OUTSIDE_BED"

        return {
            "state": state, "speed": metric_speed, "angle": torso_angle, 
            "overlap": overlap, "trunk_len": metric_trunk_len, "aspect_ratio": aspect_ratio,
            "mean_kpt_conf": mean_kpt_conf, "metric_hip": proj_hip
        }

# ---------------------------------------------------------
# 3. Main Unified Loop
# ---------------------------------------------------------
VIDEO_PATH = videopath
OUTPUT_CSV_PATH = "/workspace/data/live_telemetry_merged.csv"
SAMPLE_FPS = 5.0

print("Initialising models...")
pose_model = YOLO("yolo11m-pose.pt")
detector_model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_interval = max(1, int(round(native_fps / SAMPLE_FPS)))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video file.")

env = auto_calibrate_scene(first_frame, detector_model)
classifier = ActivityClassifier(env)
resident_tracker = ResidentTracker(max_lost_sec=3.0)

STATE_COLORS = {
    "LYING_IN_BED": (255, 105, 180), "SITTING_ON_BED": (0, 165, 255),
    "SITTING_OUTSIDE_BED": (255, 255, 0), "STANDING": (0, 255, 0),
    "WALKING": (0, 215, 255), "OUT_OF_BED": (0, 69, 255), "UNKNOWN": (128, 128, 128)
}

telemetry_records = []
display_handle = display(HTML(""), display_id=True)
print("Starting live feed and data recording...")

frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset to beginning

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            t_start = time.time()
            ts = round(frame_idx / native_fps, 2)

            results = pose_model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            vis_frame = frame.copy()

            quad_int = env.bed_quad_img.astype(np.int32)
            cv2.polylines(vis_frame, [quad_int], isClosed=True, color=(0, 200, 0), thickness=2)

            track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else np.array([])
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes.id is not None else np.array([])
            kpts_batch = results.keypoints.data.cpu().numpy() if results.boxes.id is not None else np.array([])

            match = resident_tracker.update(track_ids, boxes, kpts_batch, ts, env)

            if match is not None:
                res_id, r_box, r_kpts = match
                rx1, ry1, rx2, ry2 = map(int, r_box)
                metrics = classifier.evaluate(r_kpts, r_box, ts)
                
                current_state = metrics["state"]
                color = STATE_COLORS.get(current_state, (255, 255, 255))

                # Draw UI
                cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), color, 3)
                for kx, ky, kc in r_kpts:
                    if kc > 0.25: cv2.circle(vis_frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

                label_text = f"{current_state} (ID:{res_id})"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(vis_frame, (rx1, max(0, ry1 - th - 10)), (rx1 + tw + 6, ry1), color, -1)
                cv2.putText(vis_frame, label_text, (rx1 + 3, max(16, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

                telemetry_str = f"Time: {ts:05.1f}s | {current_state} | Angle: {metrics['angle']:.1f}° | IoP: {metrics['overlap']*100:.0f}% | L: {metrics['trunk_len']:.1f}cm | AR: {metrics['aspect_ratio']:.2f}"
                
                # Record metric data
                telemetry_records.append({
                    "frame_idx": frame_idx, "timestamp_sec": ts, "track_id": int(res_id),
                    "predicted_state": current_state, "bed_overlap_iop": round(metrics['overlap'], 4),
                    "torso_angle_deg": round(metrics['angle'], 2), "metric_speed_cms": round(metrics['speed'], 2),
                    "metric_trunk_len_cm": round(metrics['trunk_len'], 2), "bbox_aspect_ratio": round(metrics['aspect_ratio'], 3),
                    "mean_kpt_conf": round(metrics['mean_kpt_conf'], 3), "total_tracks": len(track_ids)
                })
            else:
                current_state = "UNKNOWN"
                telemetry_str = f"Time: {ts:05.1f}s | State: UNKNOWN | Resident occluded"
                telemetry_records.append({
                    "frame_idx": frame_idx, "timestamp_sec": ts, "track_id": -1, "predicted_state": "UNKNOWN",
                    "bed_overlap_iop": 0.0, "torso_angle_deg": 0.0, "metric_speed_cms": 0.0, "metric_trunk_len_cm": 0.0,
                    "bbox_aspect_ratio": 0.0, "mean_kpt_conf": 0.0, "total_tracks": len(track_ids)
                })

            cv2.rectangle(vis_frame, (10, 10), (450, 48), (0, 0, 0), -1)
            cv2.putText(vis_frame, f"STATE: {current_state}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, STATE_COLORS.get(current_state, (200, 200, 200)), 2)

            # Stream to Notebook
            _, buffer = cv2.imencode('.jpeg', vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            html_content = f'<div><p style="font-family: monospace; font-size: 13px; margin: 2px 0;"><b>{telemetry_str}</b></p><img src="data:image/jpeg;base64,{b64_str}" style="width: 700px; border-radius: 4px;" /></div>'
            display_handle.update(HTML(html_content))

            # Maintain playback speed
            elapsed = time.time() - t_start
            delay = (1.0 / SAMPLE_FPS) - elapsed
            if delay > 0: time.sleep(delay)

        frame_idx += 1

except KeyboardInterrupt:
    print("\nStream halted manually.")
finally:
    cap.release()
    if telemetry_records:
        df = pd.DataFrame(telemetry_records)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSaved {len(df)} records to {OUTPUT_CSV_PATH}")