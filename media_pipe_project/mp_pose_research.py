#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}
NUM_LM = len(LANDMARK_NAMES)

STABLE_IDXS = [LM["left_shoulder"], LM["right_shoulder"], LM["left_hip"], LM["right_hip"]]
WRIST_IDXS = [LM["left_wrist"], LM["right_wrist"]]


@dataclass
class VideoRunSummary:
    video: str
    model: str
    running_mode: str
    delegate: str

    frames_total: int
    frames_processed: int
    stride: int
    fps_src: float

    detection_rate: float
    mean_inference_ms: float
    p50_inference_ms: float
    p95_inference_ms: float
    effective_fps: float

    mean_visibility: float
    mean_presence: float
    occluded_frac_mean: float

    tracking_loss_events: int
    longest_loss_streak_frames: int

    jitter_world_m_mean: float
    jitter_world_m_p95: float
    wrist_speed_m_s_mean: float
    wrist_speed_m_s_p95: float


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def resolve_project_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def list_videos_in_dir(videos_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    return sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def find_models(model_dir: Path) -> List[Path]:
    return sorted(model_dir.glob("*.task"))


def mp_image_from_bgr(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def create_landmarker(model_path: Path,
                      delegate: str,
                      num_poses: int,
                      min_pose_detection_conf: float,
                      min_pose_presence_conf: float,
                      min_tracking_conf: float):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    if delegate == "CPU":
        d = BaseOptions.Delegate.CPU
    elif delegate == "GPU":
        d = BaseOptions.Delegate.GPU
    else:
        raise ValueError(f"Unknown delegate={delegate}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=d),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_conf,
        min_pose_presence_confidence=min_pose_presence_conf,
        min_tracking_confidence=min_tracking_conf,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def run_on_video_baseline(video_path: Path,
                          model_path: Path,
                          out_dir: Path,
                          delegate: str = "CPU",
                          stride: int = 1,
                          max_frames: int = 0,
                          num_poses: int = 1,
                          min_pose_detection_conf: float = 0.5,
                          min_pose_presence_conf: float = 0.5,
                          min_tracking_conf: float = 0.5,
                          visibility_occlusion_thr: float = 0.5) -> VideoRunSummary:

    safe_mkdir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not fps_src or fps_src <= 1e-3:
        fps_src = 30.0

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_csv_path = out_dir / "frame_metrics.csv"
    lm2d_csv_path = out_dir / "landmarks_2d.csv"
    lm3d_csv_path = out_dir / "landmarks_3d_world.csv"

    frame_f = frame_csv_path.open("w", newline="", encoding="utf-8")
    lm2d_f = lm2d_csv_path.open("w", newline="", encoding="utf-8")
    lm3d_f = lm3d_csv_path.open("w", newline="", encoding="utf-8")

    frame_w = csv.writer(frame_f)
    lm2d_w = csv.writer(lm2d_f)
    lm3d_w = csv.writer(lm3d_f)

    frame_w.writerow([
        "frame_idx", "timestamp_ms", "has_pose",
        "inference_ms",
        "mean_visibility", "mean_presence",
        "occluded_frac",
        "wrist_speed_m_s",
    ])

    lm2d_w.writerow(["frame_idx", "timestamp_ms", "landmark_idx", "landmark_name",
                     "x", "y", "z", "visibility", "presence"])
    lm3d_w.writerow(["frame_idx", "timestamp_ms", "landmark_idx", "landmark_name",
                     "x_m", "y_m", "z_m", "visibility", "presence"])

    inference_times: List[float] = []
    vis_list: List[float] = []
    pres_list: List[float] = []
    occluded_list: List[float] = []

    prev_has_pose = False
    tracking_loss_events = 0
    current_loss_streak = 0
    longest_loss_streak = 0

    prev_world: Optional[np.ndarray] = None
    prev_ts_ms: Optional[int] = None
    jitter_vals: List[float] = []
    wrist_speed_vals: List[float] = []

    processed = 0
    frame_idx = -1

    wall_t0 = time.perf_counter()

    with create_landmarker(
        model_path=model_path,
        delegate=delegate,
        num_poses=num_poses,
        min_pose_detection_conf=min_pose_detection_conf,
        min_pose_presence_conf=min_pose_presence_conf,
        min_tracking_conf=min_tracking_conf,
    ) as landmarker:

        pbar = tqdm(total=frames_total if frames_total > 0 else None,
                    desc=f"{video_path.name} | {model_path.stem} | VIDEO",
                    unit="frame")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            pbar.update(1)

            if stride > 1 and (frame_idx % stride != 0):
                continue
            processed += 1
            if max_frames > 0 and processed > max_frames:
                break

            ts_ms = int(round(frame_idx * 1000.0 / fps_src))
            mp_image = mp_image_from_bgr(frame_bgr)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, ts_ms)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            inference_times.append(infer_ms)

            has_pose = bool(getattr(result, "pose_landmarks", None)) and len(result.pose_landmarks) > 0

            if prev_has_pose and (not has_pose):
                tracking_loss_events += 1
                current_loss_streak = 1
            elif (not has_pose):
                current_loss_streak += 1
            else:
                current_loss_streak = 0
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
            prev_has_pose = has_pose

            mean_vis = 0.0
            mean_pres = 0.0
            occluded = 1.0
            wrist_speed = 0.0

            if has_pose:
                lm2d = result.pose_landmarks[0]
                lm3d = result.pose_world_landmarks[0] if getattr(result, "pose_world_landmarks", None) else None

                vis = np.array([float(getattr(lm, "visibility", 0.0)) for lm in lm2d], dtype=np.float64)
                pres = np.array([float(getattr(lm, "presence", 0.0)) for lm in lm2d], dtype=np.float64)

                mean_vis = float(np.mean(vis))
                mean_pres = float(np.mean(pres))
                occluded = float(np.mean(vis < visibility_occlusion_thr))

                for i in range(NUM_LM):
                    lm = lm2d[i]
                    lm2d_w.writerow([
                        frame_idx, ts_ms, i, LANDMARK_NAMES[i],
                        float(lm.x), float(lm.y), float(lm.z),
                        float(getattr(lm, "visibility", 0.0)),
                        float(getattr(lm, "presence", 0.0)),
                    ])

                if lm3d is not None:
                    world = np.full((NUM_LM, 3), np.nan, dtype=np.float64)
                    for i in range(NUM_LM):
                        lm = lm3d[i]
                        x, y, z = float(lm.x), float(lm.y), float(lm.z)
                        world[i] = (x, y, z)
                        lm3d_w.writerow([
                            frame_idx, ts_ms, i, LANDMARK_NAMES[i],
                            x, y, z,
                            float(getattr(lm, "visibility", 0.0)),
                            float(getattr(lm, "presence", 0.0)),
                        ])

                    if prev_world is not None and prev_ts_ms is not None:
                        dt = max(1e-3, (ts_ms - prev_ts_ms) / 1000.0)

                        if np.isfinite(world[STABLE_IDXS]).all() and np.isfinite(prev_world[STABLE_IDXS]).all():
                            diffs = world[STABLE_IDXS] - prev_world[STABLE_IDXS]
                            jitter = float(np.mean(np.linalg.norm(diffs, axis=1)))
                            jitter_vals.append(jitter)

                        if np.isfinite(world[WRIST_IDXS]).all() and np.isfinite(prev_world[WRIST_IDXS]).all():
                            wd = world[WRIST_IDXS] - prev_world[WRIST_IDXS]
                            wrist_speed = float(np.mean(np.linalg.norm(wd, axis=1)) / dt)
                            wrist_speed_vals.append(wrist_speed)

                    prev_world = world
                    prev_ts_ms = ts_ms

            vis_list.append(mean_vis)
            pres_list.append(mean_pres)
            occluded_list.append(occluded)

            frame_w.writerow([
                frame_idx, ts_ms, int(has_pose),
                float(infer_ms),
                mean_vis, mean_pres,
                occluded,
                wrist_speed,
            ])

        pbar.close()

    wall_t1 = time.perf_counter()
    cap.release()
    frame_f.close()
    lm2d_f.close()
    lm3d_f.close()

    wall_s = max(1e-6, wall_t1 - wall_t0)
    frames_processed = processed

    has_pose_count = 0
    with frame_csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            has_pose_count += int(row["has_pose"])
    detection_rate = has_pose_count / max(1, frames_processed)

    eff_fps = frames_processed / wall_s

    summary = VideoRunSummary(
        video=video_path.name,
        model=model_path.stem,
        running_mode="VIDEO",
        delegate=delegate,

        frames_total=frames_total,
        frames_processed=frames_processed,
        stride=stride,
        fps_src=float(fps_src),

        detection_rate=float(detection_rate),
        mean_inference_ms=float(np.mean(inference_times)) if inference_times else 0.0,
        p50_inference_ms=percentile(inference_times, 50),
        p95_inference_ms=percentile(inference_times, 95),
        effective_fps=float(eff_fps),

        mean_visibility=float(np.mean(vis_list)) if vis_list else 0.0,
        mean_presence=float(np.mean(pres_list)) if pres_list else 0.0,
        occluded_frac_mean=float(np.mean(occluded_list)) if occluded_list else 1.0,

        tracking_loss_events=int(tracking_loss_events),
        longest_loss_streak_frames=int(longest_loss_streak),

        jitter_world_m_mean=float(np.mean(jitter_vals)) if jitter_vals else 0.0,
        jitter_world_m_p95=percentile(jitter_vals, 95),
        wrist_speed_m_s_mean=float(np.mean(wrist_speed_vals)) if wrist_speed_vals else 0.0,
        wrist_speed_m_s_p95=percentile(wrist_speed_vals, 95),
    )

    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="../videos")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--out_dir", default="runs/run")
    ap.add_argument("--delegate", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)

    ap.add_argument("--num_poses", type=int, default=1)
    ap.add_argument("--min_pose_detection_confidence", type=float, default=0.5)
    ap.add_argument("--min_pose_presence_confidence", type=float, default=0.5)
    ap.add_argument("--min_tracking_confidence", type=float, default=0.5)
    ap.add_argument("--occlusion_thr", type=float, default=0.5)

    ap.add_argument("--models", nargs="*", default=[], help="Filter models by stem/name")
    args = ap.parse_args()

    videos_dir = resolve_project_path(args.videos_dir)
    model_dir = resolve_project_path(args.model_dir)
    out_root = resolve_project_path(args.out_dir)
    safe_mkdir(out_root)

    videos = list_videos_in_dir(videos_dir)
    if not videos:
        raise RuntimeError(f"No videos found in {videos_dir}")

    models = find_models(model_dir)
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.stem in wanted or m.name in wanted]
    if not models:
        raise RuntimeError(f"No .task models found in {model_dir}")

    summaries: List[VideoRunSummary] = []

    for video in videos:
        for model_path in models:
            run_dir = out_root / video.stem / model_path.stem / "VIDEO" / args.delegate
            summary = run_on_video_baseline(
                video_path=video,
                model_path=model_path,
                out_dir=run_dir,
                delegate=args.delegate,
                stride=int(args.stride),
                max_frames=int(args.max_frames),
                num_poses=int(args.num_poses),
                min_pose_detection_conf=float(args.min_pose_detection_confidence),
                min_pose_presence_conf=float(args.min_pose_presence_confidence),
                min_tracking_conf=float(args.min_tracking_confidence),
                visibility_occlusion_thr=float(args.occlusion_thr),
            )
            summaries.append(summary)

    summary_csv = out_root / "summary.csv"
    summary_json = out_root / "summary.json"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = list(asdict(summaries[0]).keys()) if summaries else []
        w.writerow(header)
        for s in summaries:
            d = asdict(s)
            w.writerow([d[h] for h in header])

    summary_json.write_text(json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDONE. Results in: {out_root}")
    print(f"- {summary_csv}")
    print(f"- {summary_json}")


if __name__ == "__main__":
    main()