import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

def parse_exts(value: str) -> Set[str]:
    exts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        exts.append(part.lower())
    return set(exts)

def list_files(
    dir_path: str,
    exts: Optional[Set[str]] = None,
    recursive: bool = False,
) -> List[str]:
    if recursive:
        results = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                if exts:
                    if os.path.splitext(name)[1].lower() not in exts:
                        continue
                results.append(os.path.join(root, name))
        return sorted(results)
    names = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path):
            continue
        if exts:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
        names.append(path)
    return sorted(names)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    return json.loads(content)

def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def parse_size(value: str) -> Tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError("Size must be like WIDTHxHEIGHT, e.g. 192x192.")
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must be like WIDTHxHEIGHT, e.g. 192x192.")
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Size must be integers like 192x192.") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Size must be positive.")
    return w, h

def should_process_frame(idx: int, stride: int, keep_offset: Optional[int], drop_offset: Optional[int]) -> bool:
    if stride <= 0:
        return True
    if keep_offset is not None:
        return idx % stride == keep_offset
    if drop_offset is not None:
        return idx % stride != drop_offset
    return idx % stride != 1

def is_url(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value.strip()))

def _sanitize_name(value: str, max_len: int = 80) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "video"
    return name[:max_len]

def _video_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = url.split("/")[-1].split("?")[0].split("#")[0]
    tail = _sanitize_name(tail) if tail else "video"
    return f"{tail}_{digest}"

def _url_list_digest(urls: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for url in urls:
        digest.update(url.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

def _urlfile_resume_path(out_dir: str, url_file: str) -> str:
    base = _sanitize_name(os.path.basename(url_file), max_len=48)
    suffix = hashlib.sha1(os.path.abspath(url_file).encode("utf-8")).hexdigest()[:8]
    return os.path.join(out_dir, f".extract_frames_resume_{base}_{suffix}.json")

def _load_urlfile_resume(path: str, url_digest: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return 0
    if not content:
        return 0
    try:
        if content.lstrip().startswith("{"):
            data = json.loads(content)
            if data.get("url_file_sha1") and data["url_file_sha1"] != url_digest:
                print("[resume] URL file contents changed; restarting from beginning.")
                return 0
            return max(0, int(data.get("next_index", 0)))
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("next_index="):
                line = line.split("=", 1)[1].strip()
            if line.isdigit():
                return max(0, int(line))
    except Exception:
        return 0
    return 0

def _save_urlfile_resume(
    path: str,
    url_file: str,
    url_digest: str,
    total: int,
    next_index: int,
) -> None:
    tmp_path = f"{path}.tmp"
    data = {
        "url_file": url_file,
        "url_file_sha1": url_digest,
        "total": int(total),
        "next_index": int(max(0, next_index)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)

def _run_cmd(
    cmd: List[str],
    retries: int = 2,
    backoff_sec: float = 1.0,
    capture_stdout: bool = True,
) -> Tuple[int, str, str]:
    last = None
    for attempt in range(1, retries + 2):
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if capture_stdout else None,
                stderr=subprocess.PIPE,
                text=True,
            )
            out = proc.stdout if proc.stdout is not None else ""
            err = proc.stderr if proc.stderr is not None else ""
            if proc.returncode == 0:
                return proc.returncode, out, err
            last = (proc.returncode, out, err)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Missing executable: {cmd[0]} (not in PATH).") from exc

        if attempt < retries + 1:
            time.sleep(backoff_sec * attempt)

    assert last is not None
    return last

def probe_duration_seconds(
    ffprobe: str,
    url: str,
    rw_timeout_us: int,
    user_agent: Optional[str],
    headers: Optional[str],
    retries: int,
) -> float:
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-rw_timeout",
        str(rw_timeout_us),
    ]
    if user_agent:
        cmd += ["-user_agent", user_agent]
    if headers:
        cmd += ["-headers", headers]
    cmd += [url]

    rc, out, err = _run_cmd(cmd, retries=retries, backoff_sec=1.0, capture_stdout=True)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip() or 'unknown error'}")

    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe output is not valid JSON: {exc}") from exc

    duration = None
    fmt = data.get("format") or {}
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None

    if (duration is None or duration <= 0) and isinstance(data.get("streams"), list):
        for st in data["streams"]:
            if st.get("codec_type") == "video" and "duration" in st:
                try:
                    dur2 = float(st["duration"])
                    if dur2 > 0:
                        duration = dur2
                        break
                except (TypeError, ValueError):
                    continue

    if duration is None or duration <= 0:
        raise RuntimeError("Could not read video duration from ffprobe.")
    return duration

def extract_keyframe_image(
    ffmpeg: str,
    url: str,
    ts_sec: float,
    out_path: str,
    rw_timeout_us: int,
    user_agent: Optional[str],
    headers: Optional[str],
    retries: int,
    strict_keyframe: bool,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rw_timeout",
        str(rw_timeout_us),
    ]
    if user_agent:
        cmd += ["-user_agent", user_agent]
    if headers:
        cmd += ["-headers", headers]
    cmd += ["-ss", f"{ts_sec:.3f}"]
    if strict_keyframe:
        cmd += ["-skip_frame", "nokey"]
    cmd += [
        "-i",
        url,
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-f",
        "image2",
    ]
    ext = os.path.splitext(out_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        cmd += ["-vcodec", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p", "-strict", "unofficial"]
    elif ext == ".png":
        cmd += ["-vcodec", "png"]
    cmd += [
        "-y",
        out_path,
    ]

    rc, _, err = _run_cmd(cmd, retries=retries, backoff_sec=1.0, capture_stdout=False)
    if rc != 0:
        raise RuntimeError(f"ffmpeg extract failed: {err.strip() or 'unknown error'}")

def read_url_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls

def cmd_resize_images(args: argparse.Namespace) -> None:
    exts = parse_exts(args.exts)
    img_paths = sorted(list_files(args.img_dir, exts=exts, recursive=False))
    ensure_dir(args.out_dir)
    total = len(img_paths)
    print(f"Found {total} images in {args.img_dir}")

    if args.backend == "pil":
        from PIL import Image

        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resample = interp_map.get(args.interpolation)
        if resample is None:
            raise ValueError(f"Unsupported interpolation for PIL: {args.interpolation}")
        for i, path in enumerate(img_paths, start=1):
            if args.verbose and i % 1000 == 0:
                print(f"Processed {i}/{total}")
            name = os.path.basename(path)
            out_path = os.path.join(args.out_dir, name)
            img = Image.open(path)
            img = img.resize(args.size, resample=resample)
            img.save(out_path)
    else:
        import cv2

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(args.interpolation)
        if interp is None:
            raise ValueError(f"Unsupported interpolation for OpenCV: {args.interpolation}")
        for i, path in enumerate(img_paths, start=1):
            if args.verbose and i % 1000 == 0:
                print(f"Processed {i}/{total}")
            name = os.path.basename(path)
            out_path = os.path.join(args.out_dir, name)
            img = cv2.imread(path)
            if img is None:
                if args.verbose:
                    print(f"Skipping unreadable image: {path}")
                continue
            img = cv2.resize(img, args.size, interpolation=interp)
            cv2.imwrite(out_path, img)

    print(f"Saved resized images to {args.out_dir}")

def iter_stdin_paths() -> Iterator[str]:
    if sys.stdin is None or sys.stdin.isatty():
        return
    for line in sys.stdin:
        line = line.strip()
        if line:
            yield line

def iter_videos(args: argparse.Namespace) -> Iterator[str]:
    use_stdin = False
    if args.video and "-" in args.video:
        use_stdin = True
    if args.url and "-" in args.url:
        use_stdin = True
    if args.url_file == "-":
        use_stdin = True
    elif not args.video and not args.video_dir and not args.url and not args.url_file and not sys.stdin.isatty():
        use_stdin = True

    if args.video:
        for item in args.video:
            if item == "-":
                continue
            yield item
    if args.url:
        for item in args.url:
            if item == "-":
                continue
            yield item
    if args.url_file and args.url_file != "-":
        for item in read_url_lines(args.url_file):
            yield item
    if use_stdin:
        yield from iter_stdin_paths()
    if args.video_dir:
        exts = parse_exts(args.video_exts)
        for name in os.listdir(args.video_dir):
            path = os.path.join(args.video_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            yield path

def collect_video_inputs(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    local_paths: List[str] = []
    urls: List[str] = []
    for item in iter_videos(args):
        if is_url(item):
            urls.append(item)
        else:
            local_paths.append(item)
    return local_paths, urls

def extract_remote_keyframes(
    args: argparse.Namespace,
    urls: List[str],
    resume_state: Optional[dict] = None,
) -> None:
    ensure_dir(args.out_dir)
    strict_keyframe = not args.allow_non_keyframe
    rw_timeout_us = int(max(1.0, float(args.timeout_sec)) * 1_000_000)
    total = len(urls)
    resume_start = 0
    if resume_state:
        resume_start = int(resume_state.get("start_index", 0))
        total = int(resume_state.get("total", total))
    for offset, url in enumerate(urls, start=resume_start):
        i = offset + 1
        video_name = _video_id(url)
        target_dir = args.out_dir if args.flat else os.path.join(args.out_dir, video_name)
        ensure_dir(target_dir)
        print(f"[remote {i}/{total}] Processing {url}")
        try:
            duration = probe_duration_seconds(
                ffprobe=args.ffprobe,
                url=url,
                rw_timeout_us=rw_timeout_us,
                user_agent=args.user_agent,
                headers=args.headers,
                retries=args.retries,
            )
            count = max(1, args.keyframes)
            eps = 0.10
            max_ts = max(0.0, duration - eps)
            saved = 0
            for k in range(count):
                ts = (k + 0.5) * duration / count
                ts = min(max(ts, 0.0), max_ts)
                filename = args.pattern.format(video=video_name, idx=k)
                out_path = os.path.join(target_dir, filename)
                extract_keyframe_image(
                    ffmpeg=args.ffmpeg,
                    url=url,
                    ts_sec=ts,
                    out_path=out_path,
                    rw_timeout_us=rw_timeout_us,
                    user_agent=args.user_agent,
                    headers=args.headers,
                    retries=args.retries,
                    strict_keyframe=strict_keyframe,
                )
                saved += 1
            print(f"Saved {saved} keyframes to {target_dir}")
        except Exception as exc:
            try:
                with open(os.path.join(target_dir, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(str(exc) + "\n")
            except OSError:
                pass
            print(f"[WARN] Remote extract failed for {url}: {exc}")
        finally:
            if resume_state:
                _save_urlfile_resume(
                    resume_state["path"],
                    resume_state["url_file"],
                    resume_state["url_digest"],
                    total,
                    offset + 1,
                )

def cmd_extract_frames(args: argparse.Namespace) -> None:
    local_paths, urls = collect_video_inputs(args)
    resume_state = None
    resume_completed = False
    if (
        args.url_file
        and args.url_file != "-"
        and not args.url
        and not args.video
        and not args.video_dir
        and (sys.stdin is None or sys.stdin.isatty())
    ):
        file_items = read_url_lines(args.url_file)
        local_paths = [item for item in file_items if not is_url(item)]
        urls = [item for item in file_items if is_url(item)]
        if urls:
            url_digest = _url_list_digest(urls)
            resume_path = _urlfile_resume_path(args.out_dir, args.url_file)
            start_index = _load_urlfile_resume(resume_path, url_digest)
            total = len(urls)
            if start_index >= total:
                print(f"[resume] All {total} URLs already processed for {args.url_file}.")
                urls = []
                resume_completed = True
            else:
                if start_index > 0:
                    print(f"[resume] Continue from line {start_index + 1}/{total}.")
                urls = urls[start_index:]
                resume_state = {
                    "path": resume_path,
                    "start_index": start_index,
                    "total": total,
                    "url_file": args.url_file,
                    "url_digest": url_digest,
                }
    if resume_completed and not local_paths and not urls:
        return
    if not local_paths and not urls:
        print("No videos found. Use --video/--video-dir/--url/--url-file or stdin.")
        return

    if local_paths:
        import cv2

        ensure_dir(args.out_dir)
        for i, video_path in enumerate(local_paths, start=1):
            basename = os.path.splitext(os.path.basename(video_path))[0]
            target_dir = args.out_dir if args.flat else os.path.join(args.out_dir, basename)
            ensure_dir(target_dir)

            print(f"[local {i}/{len(local_paths)}] Processing {video_path}")
            cap = cv2.VideoCapture(video_path)
            idx = 0
            saved = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if idx < args.start:
                    idx += 1
                    continue
                if args.max_frames is not None and saved >= args.max_frames:
                    break
                if should_process_frame(idx, args.frame_stride, args.keep_offset, args.drop_offset):
                    filename = args.pattern.format(video=basename, idx=idx)
                    out_path = os.path.join(target_dir, filename)
                    cv2.imwrite(out_path, frame)
                    saved += 1
                    if args.verbose and saved % 1000 == 0:
                        print(f"Saved {saved} frames from {basename}")
                idx += 1
            cap.release()
            print(f"Saved {saved} frames to {target_dir}")
            if args.delete_after:
                try:
                    os.remove(video_path)
                except OSError as exc:
                    print(f"[WARN] Failed to delete {video_path}: {exc}")

    if urls:
        if args.delete_after:
            print("[WARN] --delete-after ignored for remote URLs.")
        extract_remote_keyframes(args, urls, resume_state=resume_state)

def cmd_split_json(args: argparse.Namespace) -> None:
    data = load_json(args.json)
    if not data:
        print("Input JSON is empty.")
        return
    if args.seed is not None:
        random.seed(args.seed)
    if args.shuffle:
        random.shuffle(data)

    if args.val_ratio is not None:
        if not (0.0 <= args.val_ratio <= 1.0):
            raise ValueError("--val-ratio must be between 0 and 1.")
        ratio = args.val_ratio
        train_data = []
        val_data = []
        for item in data:
            if random.random() <= ratio:
                val_data.append(item)
            else:
                train_data.append(item)
    else:
        if args.val_count < 0:
            raise ValueError("--val-count must be >= 0.")
        if args.exact:
            val_data = data[: args.val_count]
            train_data = data[args.val_count :]
        else:
            ratio = args.val_count / max(len(data), 1)
            train_data = []
            val_data = []
            for item in data:
                if random.random() <= ratio:
                    val_data.append(item)
                else:
                    train_data.append(item)

    print(f"Train: {len(train_data)}  Val: {len(val_data)}")
    write_json(args.train_out, train_data)
    write_json(args.val_out, val_data)
    print(f"Wrote {args.train_out} and {args.val_out}")

def cmd_render_keypoints(args: argparse.Namespace) -> None:
    import cv2

    data = load_json(args.json)
    if not data:
        print("Input JSON is empty.")
        return
    if args.shuffle:
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(data)
    if args.max_samples and args.max_samples > 0:
        data = data[: args.max_samples]
    ensure_dir(args.out_dir)
    total = len(data)
    print(f"Rendering {total} images to {args.out_dir}")

    for i, item in enumerate(data, start=1):
        img_name = item.get("img_name")
        if not img_name:
            continue
        img_path = os.path.join(args.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            if args.verbose:
                print(f"Skipping missing image: {img_path}")
            continue
        h, w = img.shape[:2]

        center = item.get("center")
        if center:
            cv2.circle(
                img,
                (int(center[0] * w), int(center[1] * h)),
                args.radius,
                (0, 255, 0),
                args.thickness,
            )

        keypoints = item.get("keypoints", [])
        for k in range(len(keypoints) // 3):
            x, y, v = keypoints[k * 3 : k * 3 + 3]
            if v == 1:
                color = (255, 0, 0)
            elif v == 2:
                color = (0, 0, 255)
            elif v > 0:
                intensity = int(255 * min(max(float(v), 0.1), 1.0))
                color = (0, 0, intensity)
            else:
                continue
            cv2.circle(
                img,
                (int(x * w), int(y * h)),
                args.radius - 1 if args.radius > 1 else args.radius,
                color,
                args.thickness,
            )

        if args.draw_other:
            for other_center in item.get("other_centers", []):
                cv2.circle(
                    img,
                    (int(other_center[0] * w), int(other_center[1] * h)),
                    args.radius,
                    (0, 255, 255),
                    args.thickness,
                )
            for other_keypoints in item.get("other_keypoints", []):
                for k in other_keypoints:
                    cv2.circle(
                        img,
                        (int(k[0] * w), int(k[1] * h)),
                        args.radius - 1 if args.radius > 1 else args.radius,
                        (255, 255, 0),
                        args.thickness,
                    )

        out_path = os.path.join(args.out_dir, img_name)
        cv2.imwrite(out_path, img)
        if args.verbose and i % 1000 == 0:
            print(f"Rendered {i}/{total}")

def cmd_find_unlabeled(args: argparse.Namespace) -> None:
    exts = parse_exts(args.exts)
    img_paths = list_files(args.img_dir, exts=exts, recursive=False)
    img_names = [os.path.basename(p) for p in img_paths]
    data = load_json(args.json)
    data_names = set(item.get("img_name") for item in data if "img_name" in item)

    missing = [name for name in img_names if name not in data_names]
    print(f"Found {len(missing)} unlabeled images.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for name in missing:
                f.write(name + "\n")
        print(f"Wrote {args.out}")
    else:
        for name in missing:
            print(name)

def cmd_prune_by_show(args: argparse.Namespace) -> None:
    img_names = set(os.listdir(args.img_dir))
    show_names = set(os.listdir(args.show_dir))
    data = load_json(args.json)

    new_data = []
    to_move = []
    for item in data:
        name = item.get("img_name")
        if not name:
            continue
        if name in show_names:
            new_data.append(item)
        else:
            if name in img_names:
                to_move.append(name)

    print(f"Labels kept: {len(new_data)}")
    print(f"Images to move: {len(to_move)}")
    if not args.apply:
        print("Dry-run: no files moved and no JSON written. Use --apply to execute.")
        return

    ensure_dir(args.trash_dir)
    for name in to_move:
        src = os.path.join(args.img_dir, name)
        dst = os.path.join(args.trash_dir, name)
        if os.path.exists(dst) and not args.overwrite:
            if args.verbose:
                print(f"Skipping existing file: {dst}")
            continue
        shutil.move(src, dst)

    json_out = args.json_out or args.json
    write_json(json_out, new_data)
    print(f"Wrote {json_out}")

def cmd_collect_labeled(args: argparse.Namespace) -> None:
    img_exts = parse_exts(args.img_exts)
    label_exts = parse_exts(args.label_exts)
    imgs = list_files(args.img_dir, exts=img_exts, recursive=args.recursive)
    labels = list_files(args.label_dir, exts=label_exts, recursive=args.recursive)
    label_basenames = set(os.path.basename(p) for p in labels)

    print(f"Images scanned: {len(imgs)}")
    print(f"Label files scanned: {len(labels)}")

    ensure_dir(args.out_dir)
    moved = 0
    for path in imgs:
        name = os.path.basename(path)
        if name not in label_basenames:
            continue
        dst = os.path.join(args.out_dir, name)
        if os.path.exists(dst) and not args.overwrite:
            if args.verbose:
                print(f"Skipping existing file: {dst}")
            continue
        if not args.apply:
            moved += 1
            continue
        if args.copy:
            shutil.copy2(path, dst)
        else:
            shutil.move(path, dst)
        moved += 1

    if not args.apply:
        print(f"Dry-run: would move/copy {moved} images. Use --apply to execute.")
    else:
        print(f"Moved/copied {moved} images to {args.out_dir}")

def cmd_collect_pairs(args: argparse.Namespace) -> None:
    def first_stem(name: str) -> str:
        return name.split(".")[0]

    ref_names = []
    for name in os.listdir(args.ref_dir):
        path = os.path.join(args.ref_dir, name)
        if os.path.isfile(path):
            ref_names.append(name)
    stems = set(first_stem(name) for name in ref_names)
    print(f"Reference stems: {len(stems)}")

    ensure_dir(args.out_dir)
    moved = 0

    for name in os.listdir(args.img_dir):
        src = os.path.join(args.img_dir, name)
        if not os.path.isfile(src):
            continue
        if first_stem(name) not in stems:
            continue
        dst = os.path.join(args.out_dir, name)
        if os.path.exists(dst) and not args.overwrite:
            if args.verbose:
                print(f"Skipping existing file: {dst}")
            continue
        if not args.apply:
            moved += 1
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        moved += 1

    for name in os.listdir(args.label_dir):
        src = os.path.join(args.label_dir, name)
        if not os.path.isfile(src):
            continue
        if first_stem(name) not in stems:
            continue
        dst = os.path.join(args.out_dir, name)
        if os.path.exists(dst) and not args.overwrite:
            if args.verbose:
                print(f"Skipping existing file: {dst}")
            continue
        if not args.apply:
            moved += 1
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        moved += 1

    if not args.apply:
        print(f"Dry-run: would move/copy {moved} files. Use --apply to execute.")
    else:
        print(f"Moved/copied {moved} files to {args.out_dir}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-tools",
        description="Unified CLI for dataset preparation utilities.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_resize = subparsers.add_parser("resize-images", help="Resize images to a fixed size.")
    p_resize.add_argument("--img-dir", required=True, help="Input image directory.")
    p_resize.add_argument("--out-dir", required=True, help="Output directory for resized images.")
    p_resize.add_argument("--resize", dest="size", type=parse_size, required=True, help="Target size, e.g. 192x192.")
    p_resize.add_argument(
        "--backend",
        choices=["pil", "cv2"],
        default="pil",
        help="Image backend to use.",
    )
    p_resize.add_argument(
        "--interpolation",
        default="bilinear",
        help="Interpolation method (pil: nearest|bilinear|bicubic|lanczos; cv2: nearest|bilinear|bicubic|area|lanczos).",
    )
    p_resize.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to include.",
    )
    p_resize.set_defaults(func=cmd_resize_images)

    p_extract = subparsers.add_parser("extract-frames", help="Extract frames from videos.")
    p_extract.add_argument(
        "--video",
        action="append",
        help="Video path or URL (repeatable). Use '-' to read paths from stdin.",
    )
    p_extract.add_argument("--video-dir", help="Directory containing videos.")
    p_extract.add_argument(
        "--url",
        action="append",
        help="Remote video URL (repeatable). Use '-' to read URLs from stdin.",
    )
    p_extract.add_argument(
        "--url-file",
        help="Text file with one URL per line. Use '-' to read from stdin.",
    )
    p_extract.add_argument("--out-dir", required=True, help="Output directory for frames.")
    p_extract.add_argument("--frame-stride", type=int, default=4, help="Frame stride (N).")
    p_extract.add_argument(
        "--keep-offset",
        type=int,
        default=None,
        help="Only keep frames where idx % stride == offset.",
    )
    p_extract.add_argument(
        "--drop-offset",
        type=int,
        default=None,
        help="Drop frames where idx % stride == offset.",
    )
    p_extract.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this frame index.",
    )
    p_extract.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to save per video.",
    )
    p_extract.add_argument(
        "--keyframes",
        type=int,
        default=8,
        help="Keyframes per remote video (URL inputs only).",
    )
    p_extract.add_argument(
        "--pattern",
        default="{video}_{idx}.jpg",
        help="Output filename pattern (use {video} and {idx}).",
    )
    p_extract.add_argument(
        "--flat",
        action="store_true",
        help="Save all frames directly in --out-dir (no per-video subdir).",
    )
    p_extract.add_argument(
        "--video-exts",
        default=".mp4,.avi,.mov,.mkv,.flv",
        help="Comma-separated video extensions to include.",
    )
    p_extract.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete video file after extracting frames.",
    )
    p_extract.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path.")
    p_extract.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable path.")
    p_extract.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="Network timeout in seconds for remote URLs.",
    )
    p_extract.add_argument("--retries", type=int, default=2, help="Retry count for ffmpeg/ffprobe.")
    p_extract.add_argument("--user-agent", type=str, default=None, help="Custom User-Agent for remote URLs.")
    p_extract.add_argument(
        "--headers",
        type=str,
        default=None,
        help="Custom HTTP headers for remote URLs.",
    )
    p_extract.add_argument(
        "--allow-non-keyframe",
        action="store_true",
        help="Allow non-keyframe extraction for remote URLs.",
    )
    p_extract.set_defaults(func=cmd_extract_frames)

    p_split = subparsers.add_parser("split-json", help="Split JSON annotations into train/val.")
    p_split.add_argument("--json", required=True, help="Input JSON file.")
    p_split.add_argument("--train-out", required=True, help="Output train JSON path.")
    p_split.add_argument("--val-out", required=True, help="Output val JSON path.")
    group = p_split.add_mutually_exclusive_group(required=True)
    group.add_argument("--val-count", type=int, help="Target number of validation samples.")
    group.add_argument("--val-ratio", type=float, help="Validation ratio (0-1).")
    p_split.add_argument("--seed", type=int, default=None, help="Random seed.")
    p_split.add_argument("--shuffle", action="store_true", help="Shuffle before splitting.")
    p_split.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle.")
    p_split.set_defaults(shuffle=True)
    p_split.add_argument("--exact", action="store_true", help="Use exact --val-count after shuffle.")
    p_split.set_defaults(func=cmd_split_json)

    p_render = subparsers.add_parser("render-keypoints", help="Render keypoints on images.")
    p_render.add_argument("--json", required=True, help="Input JSON file.")
    p_render.add_argument("--img-dir", required=True, help="Image directory.")
    p_render.add_argument("--out-dir", required=True, help="Output directory.")
    p_render.add_argument("--draw-other", action="store_true", help="Draw other centers/keypoints.")
    p_render.add_argument("--no-draw-other", dest="draw_other", action="store_false", help="Skip other centers/keypoints.")
    p_render.add_argument("--radius", type=int, default=4, help="Circle radius.")
    p_render.add_argument("--thickness", type=int, default=2, help="Circle thickness.")
    p_render.add_argument("-m", "--max-samples", type=int, default=0, help="Max images to render (0 = no limit).")
    p_render.add_argument("--shuffle", action="store_true", help="Shuffle before sampling.")
    p_render.add_argument("--seed", type=int, default=None, help="Random seed for shuffle.")
    p_render.set_defaults(func=cmd_render_keypoints, draw_other=True)

    p_find = subparsers.add_parser("find-unlabeled", help="Find images without labels.")
    p_find.add_argument("--img-dir", required=True, help="Image directory.")
    p_find.add_argument("--json", required=True, help="Input JSON file.")
    p_find.add_argument("--out", help="Output text file for missing images.")
    p_find.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to include.",
    )
    p_find.set_defaults(func=cmd_find_unlabeled)

    p_prune = subparsers.add_parser("prune-by-show", help="Filter labels by show dir and move images.")
    p_prune.add_argument("--img-dir", required=True, help="Source image directory.")
    p_prune.add_argument("--show-dir", required=True, help="Show directory containing kept images.")
    p_prune.add_argument("--json", required=True, help="Input JSON file.")
    p_prune.add_argument("--json-out", help="Output JSON path (default: overwrite input).")
    p_prune.add_argument("--trash-dir", default="del", help="Directory to move removed images.")
    p_prune.add_argument("--apply", action="store_true", help="Execute changes (move + write JSON).")
    p_prune.add_argument("--overwrite", action="store_true", help="Overwrite existing files in trash.")
    p_prune.set_defaults(func=cmd_prune_by_show)

    p_collect = subparsers.add_parser("collect-labeled", help="Collect images that have matching label files.")
    p_collect.add_argument("--img-dir", required=True, help="Directory of source images.")
    p_collect.add_argument("--label-dir", required=True, help="Directory of label/reference files.")
    p_collect.add_argument("--out-dir", required=True, help="Output directory for collected images.")
    p_collect.add_argument("--apply", action="store_true", help="Execute changes.")
    p_collect.add_argument("--copy", action="store_true", help="Copy instead of move.")
    p_collect.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p_collect.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    p_collect.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level directory.",
    )
    p_collect.add_argument(
        "--img-exts",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to include.",
    )
    p_collect.add_argument(
        "--label-exts",
        default=".jpg,.jpeg,.png",
        help="Comma-separated label/reference extensions to include.",
    )
    p_collect.set_defaults(func=cmd_collect_labeled, recursive=True)

    p_pairs = subparsers.add_parser("collect-pairs", help="Collect image/label pairs by reference names.")
    p_pairs.add_argument("--ref-dir", required=True, help="Directory with reference files (basename used).")
    p_pairs.add_argument("--img-dir", required=True, help="Directory with images.")
    p_pairs.add_argument("--label-dir", required=True, help="Directory with labels.")
    p_pairs.add_argument("--out-dir", required=True, help="Output directory for collected pairs.")
    p_pairs.add_argument("--apply", action="store_true", help="Execute changes.")
    p_pairs.add_argument("--copy", action="store_true", help="Copy instead of move.")
    p_pairs.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p_pairs.set_defaults(func=cmd_collect_pairs)

    return parser

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract-frames":
        if args.keep_offset is not None and args.drop_offset is not None:
            parser.error("Use only one of --keep-offset or --drop-offset.")
        if args.frame_stride <= 0:
            parser.error("--frame-stride must be a positive integer.")
        if args.keep_offset is not None and args.keep_offset >= args.frame_stride:
            parser.error("--keep-offset must be smaller than --frame-stride.")
        if args.drop_offset is not None and args.drop_offset >= args.frame_stride:
            parser.error("--drop-offset must be smaller than --frame-stride.")
        if args.keyframes <= 0:
            parser.error("--keyframes must be a positive integer.")
        if args.timeout_sec <= 0:
            parser.error("--timeout-sec must be a positive number.")
    args.func(args)

if __name__ == "__main__":
    main()
