#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# Константы landmarks
# =========================
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

POSE_CONNECTIONS = [
    (LM["left_shoulder"], LM["right_shoulder"]),
    (LM["left_shoulder"], LM["left_hip"]),
    (LM["right_shoulder"], LM["right_hip"]),
    (LM["left_hip"], LM["right_hip"]),
    (LM["left_shoulder"], LM["left_elbow"]),
    (LM["left_elbow"], LM["left_wrist"]),
    (LM["left_wrist"], LM["left_thumb"]),
    (LM["left_wrist"], LM["left_index"]),
    (LM["left_wrist"], LM["left_pinky"]),
    (LM["right_shoulder"], LM["right_elbow"]),
    (LM["right_elbow"], LM["right_wrist"]),
    (LM["right_wrist"], LM["right_thumb"]),
    (LM["right_wrist"], LM["right_index"]),
    (LM["right_wrist"], LM["right_pinky"]),
]

HIP_IDXS = [LM["left_hip"], LM["right_hip"]]


# =========================
# Утилиты
# =========================
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_from_list(items: List[Path], title: str, display_func: Callable[[Path], str] = lambda p: p.name) -> Path:
    if not items:
        raise RuntimeError(f"Список пуст: {title}")

    print(f"\n{title}")
    for i, item in enumerate(items, start=1):
        print(f"{i}) {display_func(item)}")
    while True:
        s = input("Выбери номер: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("Некорректный ввод, попробуй ещё раз.")


def list_sets(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    # compare_out / analysis_out не считаем наборами результатов
    bad_names = {"compare_out", "advanced_analysis_out", "table_analysis"}
    return sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name not in bad_names])


def list_leaf_dirs(run_set: Path) -> List[Path]:
    leafs = []
    for p in run_set.rglob("landmarks_3d_world_torso_arms.csv"):
        leafs.append(p.parent)
    for p in run_set.rglob("landmarks_3d_world.csv"):
        leafs.append(p.parent)
    return sorted(set(leafs))


def tags_from_leaf(leaf_dir: Path, set_root: Path) -> Tuple[str, str, str, str]:
    rel = leaf_dir.relative_to(set_root)
    parts = rel.parts
    video = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    mode = parts[2] if len(parts) > 2 else ""
    delegate = parts[3] if len(parts) > 3 else ""
    return video, model, mode, delegate


def short_model_name(model_name: str) -> str:
    return model_name.replace("pose_landmarker_", "").replace("pose_landmarker", "").strip("_")


def short_video_name(video_name: str) -> str:
    return Path(video_name).stem


def short_label(video_name: str, model_name: str) -> str:
    return f"{short_video_name(video_name)}|{short_model_name(model_name)}"


# =========================
# Чтение summary.csv
# =========================
def load_summary_csv(path: Path) -> List[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(x, default=np.nan) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def collect_metrics_from_summary(runs_root: Path) -> List[dict]:
    """
    Собирает метрики только из summary.csv по всем наборам в runs/*
    """
    rows = []

    for run_set in list_sets(runs_root):
        summary_path = run_set / "summary.csv"
        if not summary_path.exists():
            print(f"[skip] summary.csv not found in {run_set}")
            continue

        summary_rows = load_summary_csv(summary_path)

        for r in summary_rows:
            video = r.get("video", "")
            model = r.get("model", "")
            delegate = r.get("delegate", "")
            mode = r.get("running_mode", r.get("mode", ""))

            # Универсально: для baseline и opt
            mean_visibility = r.get("mean_visibility_keep", r.get("mean_visibility", ""))
            mean_presence = r.get("mean_presence_keep", r.get("mean_presence", ""))
            jitter_mean = r.get("jitter_world_m_mean", "")

            rows.append({
                "set": run_set.name,
                "video": video,
                "model": model,
                "mode": mode,
                "delegate": delegate,
                "fps_mean": to_float(r.get("effective_fps")),
                "visibility_mean": to_float(mean_visibility),
                "presence_mean": to_float(mean_presence),
                "jitter_mean": to_float(jitter_mean),
            })

    return rows


# =========================
# Чтение landmark csv для playback
# =========================
def load_landmarks_3d_csv(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    frame_ids_sorted = sorted({int(r["frame_idx"]) for r in rows})
    landmark_ids_sorted = sorted({int(r["landmark_idx"]) for r in rows})

    fi_to_idx = {fi: i for i, fi in enumerate(frame_ids_sorted)}
    li_to_idx = {li: j for j, li in enumerate(landmark_ids_sorted)}

    ts_ms = np.zeros(len(frame_ids_sorted), dtype=np.int64)
    coords = np.full((len(frame_ids_sorted), len(landmark_ids_sorted), 3), np.nan, dtype=np.float64)
    vis = np.full((len(frame_ids_sorted), len(landmark_ids_sorted)), np.nan, dtype=np.float64)
    pres = np.full((len(frame_ids_sorted), len(landmark_ids_sorted)), np.nan, dtype=np.float64)

    # поддержка обеих схем: x_m/y_m/z_m или x/y/z
    for r in rows:
        i = fi_to_idx[int(r["frame_idx"])]
        j = li_to_idx[int(r["landmark_idx"])]
        ts_ms[i] = int(r["timestamp_ms"])

        x_key = "x_m" if "x_m" in r else "x"
        y_key = "y_m" if "y_m" in r else "y"
        z_key = "z_m" if "z_m" in r else "z"

        coords[i, j, 0] = float(r[x_key])
        coords[i, j, 1] = float(r[y_key])
        coords[i, j, 2] = float(r[z_key])
        vis[i, j] = float(r.get("visibility", "nan"))
        pres[i, j] = float(r.get("presence", "nan"))

    return (
        np.array(frame_ids_sorted, dtype=np.int64),
        ts_ms,
        np.array(landmark_ids_sorted, dtype=np.int64),
        coords,
        vis,
        pres,
    )


# =========================
# Графики
# =========================
def save_bar(labels: List[str], values: List[float], title: str, ylabel: str, out_path: Path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_scatter(x: List[float], y: List[float], labels: List[str],
                 title: str, xlabel: str, ylabel: str, out_path: Path):
    if not x or not y:
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def grouped_mean(rows: List[dict], key_x: str, key_y: str) -> Tuple[List[str], List[float]]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        val = r[key_y]
        if not np.isfinite(val):
            continue
        groups.setdefault(r[key_x], []).append(val)

    labels = sorted(groups.keys())
    values = [float(np.mean(groups[k])) for k in labels]
    return labels, values


def make_graphs(runs_root: Path, out_dir: Path):
    safe_mkdir(out_dir)
    plots_dir = out_dir / "plots"
    safe_mkdir(plots_dir)

    rows = collect_metrics_from_summary(runs_root)
    if not rows:
        raise RuntimeError("Не найдено ни одного summary.csv в runs/*")

    # 1) средний fps по каждой модели и каждому видео
    labels = [short_label(r["video"], r["model"]) for r in rows]
    values = [r["fps_mean"] for r in rows]
    save_bar(
        labels, values,
        "Mean FPS by each video/model",
        "FPS",
        plots_dir / "fps_by_video_and_model.png"
    )

    # 2) средний fps по каждой модели, среднее для всех видео
    labels, values = grouped_mean(rows, "model", "fps_mean")
    labels = [short_model_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean FPS by model (mean over all videos)",
        "FPS",
        plots_dir / "fps_by_model_mean_all_videos.png"
    )

    # 3) средний fps по каждому видео, среднее для всех моделей
    labels, values = grouped_mean(rows, "video", "fps_mean")
    labels = [short_video_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean FPS by video (mean over all models)",
        "FPS",
        plots_dir / "fps_by_video_mean_all_models.png"
    )

    # 4) visibility vs модель
    labels, values = grouped_mean(rows, "model", "visibility_mean")
    labels = [short_model_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean visibility vs model",
        "Visibility",
        plots_dir / "visibility_vs_model.png"
    )

    # 5) visibility vs видео
    labels, values = grouped_mean(rows, "video", "visibility_mean")
    labels = [short_video_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean visibility vs video",
        "Visibility",
        plots_dir / "visibility_vs_video.png"
    )

    # 6) presence vs модель
    labels, values = grouped_mean(rows, "model", "presence_mean")
    labels = [short_model_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean presence vs model",
        "Presence",
        plots_dir / "presence_vs_model.png"
    )

    # 7) presence vs видео
    labels, values = grouped_mean(rows, "video", "presence_mean")
    labels = [short_video_name(x) for x in labels]
    save_bar(
        labels, values,
        "Mean presence vs video",
        "Presence",
        plots_dir / "presence_vs_video.png"
    )

    # 8) jitter vs модель
    labels, values = grouped_mean(rows, "model", "jitter_mean")
    labels = [short_model_name(x) for x in labels]
    save_bar(
        labels, values,
        "Jitter vs model",
        "Jitter",
        plots_dir / "jitter_vs_model.png"
    )

    # 9) jitter vs видео
    labels, values = grouped_mean(rows, "video", "jitter_mean")
    labels = [short_video_name(x) for x in labels]
    save_bar(
        labels, values,
        "Jitter vs video",
        "Jitter",
        plots_dir / "jitter_vs_video.png"
    )

    # 10) jitter vs mean presence
    save_scatter(
        [r["presence_mean"] for r in rows if np.isfinite(r["presence_mean"]) and np.isfinite(r["jitter_mean"])],
        [r["jitter_mean"] for r in rows if np.isfinite(r["presence_mean"]) and np.isfinite(r["jitter_mean"])],
        [short_label(r["video"], r["model"]) for r in rows if np.isfinite(r["presence_mean"]) and np.isfinite(r["jitter_mean"])],
        "Jitter vs mean presence",
        "Mean presence",
        "Jitter",
        plots_dir / "jitter_vs_mean_presence.png"
    )

    # Сводная таблица
    csv_out = out_dir / "metrics_summary_from_summary_csv.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "set", "video", "model", "mode", "delegate",
                "fps_mean", "visibility_mean", "presence_mean", "jitter_mean"
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nГотово.")
    print(f"Таблица: {csv_out}")
    print(f"Графики: {plots_dir}")


# =========================
# Видео + 3D представление
# =========================
def find_matching_video(video_name: str, videos_root: Path) -> Path:
    stem = Path(video_name).stem
    candidates = list(videos_root.glob(f"{stem}.*"))
    if not candidates:
        raise RuntimeError(f"Не найдено видео для stem={stem} в {videos_root}")
    return candidates[0]


def load_leaf_for_playback(leaf_dir: Path):
    csv_path_1 = leaf_dir / "landmarks_3d_world_torso_arms.csv"
    csv_path_2 = leaf_dir / "landmarks_3d_world.csv"

    if csv_path_1.exists():
        csv_path = csv_path_1
    elif csv_path_2.exists():
        csv_path = csv_path_2
    else:
        raise RuntimeError(f"Нет 3D csv в {leaf_dir}")

    frame_ids, ts_ms, landmark_ids, coords, vis, pres = load_landmarks_3d_csv(csv_path)
    return frame_ids, ts_ms, landmark_ids, coords, vis, pres


def play_video_and_3d(runs_root: Path, videos_root: Path):
    run_set = choose_from_list(list_sets(runs_root), "Выбери набор results (например run или run_opt):")
    leaf = choose_from_list(list_leaf_dirs(run_set), f"Выбери прогон из {run_set.name}:", display_func=lambda p: short_label(*tags_from_leaf(p, run_set)[:2]))
    video_name, model, mode, delegate = tags_from_leaf(leaf, run_set)

    video_path = find_matching_video(video_name, videos_root)
    frame_ids, ts_ms, landmark_ids, coords, vis, pres = load_leaf_for_playback(leaf)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    li_to_j = {li: j for j, li in enumerate(landmark_ids.tolist())}

    # Центрирование по hips
    coords_draw = coords.copy()
    for t in range(coords_draw.shape[0]):
        hip_points = []
        for hip_idx in HIP_IDXS:
            if hip_idx in li_to_j:
                j = li_to_j[hip_idx]
                p = coords_draw[t, j]
                if np.isfinite(p).all():
                    hip_points.append(p)
        if hip_points:
            center = np.mean(np.array(hip_points), axis=0)
            coords_draw[t] = coords_draw[t] - center

    pts_all = coords_draw.reshape(-1, 3)
    pts_all = pts_all[np.isfinite(pts_all).all(axis=1)]
    if len(pts_all) == 0:
        raise RuntimeError("Нет валидных 3D-координат для отрисовки.")

    mins = pts_all.min(axis=0)
    maxs = pts_all.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.15 * span
    mins -= pad
    maxs += pad

    fig = plt.figure(figsize=(15, 7))
    ax_video = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    ax_video.set_title(f"Video: {short_video_name(video_name)}")
    ax_video.axis("off")

    ax_3d.set_title(f"3D: {short_model_name(model)}")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_xlim(mins[0], maxs[0])
    ax_3d.set_ylim(mins[1], maxs[1])
    ax_3d.set_zlim(mins[2], maxs[2])

    # Легенда цветов
    ax_3d.text2D(0.02, 0.08, "blue = visibility", transform=ax_3d.transAxes, color="blue", fontsize=9)
    ax_3d.text2D(0.02, 0.04, "red = presence", transform=ax_3d.transAxes, color="red", fontsize=9)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[0]))
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_artist = ax_video.imshow(frame_rgb)

    scat = ax_3d.scatter([], [], [], s=20)
    line_artists = []
    for a, b in POSE_CONNECTIONS:
        ln, = ax_3d.plot([], [], [], linewidth=2)
        line_artists.append((ln, a, b))

    vis_texts = []
    pres_texts = []

    info_text = ax_3d.text2D(0.02, 0.98, "", transform=ax_3d.transAxes, va="top")

    dts = np.diff(ts_ms.astype(np.float64)) / 1000.0
    if len(dts) == 0:
        dts = np.array([1 / 30], dtype=np.float64)
    frame_delays = np.concatenate([dts, [dts[-1]]], axis=0)

    t_wall_prev = None

    def get_video_frame(frame_number: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frm = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    def update(k: int):
        nonlocal t_wall_prev

        if t_wall_prev is None:
            t_wall_prev = time.perf_counter()
        else:
            target = frame_delays[k]
            now = time.perf_counter()
            elapsed = now - t_wall_prev
            if elapsed < target:
                time.sleep(target - elapsed)
            t_wall_prev = time.perf_counter()

        target_frame = int(frame_ids[k])

        frm = get_video_frame(target_frame)
        if frm is not None:
            img_artist.set_data(frm)

        pts = coords_draw[k]
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for txt in vis_texts:
            txt.remove()
        for txt in pres_texts:
            txt.remove()
        vis_texts.clear()
        pres_texts.clear()

        for ln, a, b in line_artists:
            if a in li_to_j and b in li_to_j:
                ja = li_to_j[a]
                jb = li_to_j[b]
                pa = pts[ja]
                pb = pts[jb]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    ln.set_3d_properties([pa[2], pb[2]])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
            else:
                ln.set_data([], [])
                ln.set_3d_properties([])

        # visibility слева, presence справа
        for li, j in li_to_j.items():
            p = pts[j]
            if not np.isfinite(p).all():
                continue

            v = vis[k, j]
            pr = pres[k, j]

            t_vis = ax_3d.text(
                p[0] - 0.015, p[1], p[2],
                f"v:{v:.2f}",
                fontsize=6,
                color="blue",
                ha="right"
            )
            t_pr = ax_3d.text(
                p[0] + 0.015, p[1], p[2],
                f"p:{pr:.2f}",
                fontsize=6,
                color="red",
                ha="left"
            )

            vis_texts.append(t_vis)
            pres_texts.append(t_pr)

        info_text.set_text(
            f"frame={target_frame} | t={ts_ms[k]} ms\n"
            f"video={short_video_name(video_name)} | model={short_model_name(model)}"
        )

        return [img_artist, scat, info_text] + [ln for ln, _, _ in line_artists] + vis_texts + pres_texts

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()
    cap.release()


# =========================
# Главное меню
# =========================
def main():
    runs_root = Path("runs").resolve()
    videos_root = Path("../videos").resolve()

    print("Выбери режим:")
    print("1) Создать графики")
    print("2) Запустить видео + 3D представление")

    choice = input("Номер: ").strip()

    if choice == "1":
        out_dir = runs_root / "advanced_analysis_out"
        make_graphs(runs_root, out_dir)
    elif choice == "2":
        play_video_and_3d(runs_root, videos_root)
    else:
        print("Некорректный выбор.")


if __name__ == "__main__":
    main()