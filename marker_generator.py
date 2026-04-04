"""
Quran Marker Generator — Page & Verse Boundaries

Generates JSON marker files indicating page-split and/or verse-split
positions (in seconds) for each surah MP3 input. Supports batch
processing across multiple reciters.

Directory layout:
  recitations/<reciter>/001.mp3 … 114.mp3    (input)
  timestamps/<reciter>/001.json … 114.json   (cached transcriptions)
  markers/<reciter>/pages/001.json …         (page boundary markers)
  markers/<reciter>/verses/001.json …        (verse boundary markers)

Output format — pages (N-1 markers for N pages):
  {"markers": [{"time": 172.34}, {"time": 335.12}, ...]}

Output format — verses (one entry per verse):
  {"markers": [{"verse": 0, "time": 0.0}, {"verse": 1, "time": 5.23}, ...]}

Usage:
  uv run python marker_generator.py                               # all reciters, both modes
  uv run python marker_generator.py --reciter alafasy              # one reciter
  uv run python marker_generator.py --reciter alafasy,luhaidan     # multiple
  uv run python marker_generator.py --mode pages                   # pages only
  uv run python marker_generator.py --mode verses                  # verses only
  uv run python marker_generator.py --surah 7-16                   # specific surahs
  uv run python marker_generator.py --model base                   # whisper model
"""

import argparse
import array
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from aligner import transcribe_surah, align_surah

BASE_DIR = Path(__file__).parent
RECITATIONS_DIR = BASE_DIR / "recitations"
TIMESTAMPS_DIR = BASE_DIR / "timestamps"
MARKERS_DIR = BASE_DIR / "markers"
QURAN_JSON = BASE_DIR / "quran.json"


# ---------------------------------------------------------------------------
# Quran data helpers (from splitter.py, standalone)
# ---------------------------------------------------------------------------

def load_quran_data() -> dict:
    """Load quran.json and build surah -> [verse_dicts] mapping."""
    with open(QURAN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    surah_verses: dict[int, list[dict]] = {}
    for _page_key, verses in data.items():
        for v in verses:
            ch = v["chapter"]
            if ch not in surah_verses:
                surah_verses[ch] = []
            surah_verses[ch].append(v)

    for ch in surah_verses:
        surah_verses[ch].sort(key=lambda x: x["verse"])

    # Remove duplicates (quran.json may have dupes from page grouping)
    for ch in surah_verses:
        seen: set[int] = set()
        unique: list[dict] = []
        for v in surah_verses[ch]:
            if v["verse"] not in seen:
                seen.add(v["verse"])
                unique.append(v)
        surah_verses[ch] = unique

    return surah_verses


def get_surah_page_info(surah_verses: dict) -> dict:
    """Build surah -> {first_page, pages, num_pages}."""
    info = {}
    for ch, verses in surah_verses.items():
        pages = sorted(set(v["page"] for v in verses))
        info[ch] = {
            "first_page": pages[0],
            "pages": pages,
            "num_pages": len(pages),
        }
    return info


def get_audio_duration_ms(audio_path: str) -> int | None:
    """Get audio duration in milliseconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return int(float(result.stdout.strip()) * 1000)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Split-point computation & energy refinement (from splitter.py)
# ---------------------------------------------------------------------------

def _compute_split_points(segments: list[dict], audio_duration_ms: int) -> list[int]:
    """Compute initial split points from segment timestamps (midpoints)."""
    n = len(segments)
    points = [0] * (n + 1)
    points[0] = 0
    points[n] = audio_duration_ms

    for i in range(1, n):
        prev_end = min(segments[i - 1].get("end_ms") or 0, audio_duration_ms)
        curr_start = min(segments[i].get("start_ms") or prev_end, audio_duration_ms)
        points[i] = (prev_end + curr_start) // 2

    for i in range(1, n + 1):
        points[i] = max(points[i], points[i - 1])
        points[i] = min(points[i], audio_duration_ms)

    return points


def _load_raw_pcm(audio_path: str) -> tuple[array.array, int]:
    """Convert audio to raw 16kHz mono PCM. Returns (samples, sample_rate)."""
    raw_path = tempfile.mktemp(suffix=".raw")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1", "-f", "s16le", raw_path],
            capture_output=True,
        )
        with open(raw_path, "rb") as f:
            raw_data = f.read()
    finally:
        if os.path.exists(raw_path):
            os.unlink(raw_path)
    return array.array("h", raw_data), 16000


def _refine_page_splits_with_energy(
    audio_path: str,
    pages: list[dict],
    split_points: list[int],
    audio_duration_ms: int,
) -> list[int]:
    """
    Refine ALL page boundary split points using audio energy analysis.

    Searches for natural silence gaps near each boundary and places the
    split at the end of the best silence region.
    """
    if len(pages) <= 1:
        return split_points

    n_boundaries = len(pages) - 1
    print(f"  Refining {n_boundaries} page boundaries with audio energy analysis...")

    samples, sample_rate = _load_raw_pcm(audio_path)
    frame_ms = 10
    frame_samples = int(frame_ms * sample_rate / 1000)
    smooth_radius = 5
    total_samples = len(samples)
    MIN_SILENCE_MS = 200

    def _find_silences_in_window(search_lo_ms, search_hi_ms):
        search_lo = int(search_lo_ms * sample_rate / 1000)
        search_hi = min(total_samples, int(search_hi_ms * sample_rate / 1000))

        energies = []
        for pos in range(search_lo, search_hi - frame_samples, frame_samples):
            frame = samples[pos:pos + frame_samples]
            rms = (sum(s * s for s in frame) / frame_samples) ** 0.5
            energies.append(rms)

        if len(energies) < smooth_radius * 2 + 1:
            return [], energies, []

        smoothed = []
        for j in range(len(energies)):
            lo = max(0, j - smooth_radius)
            hi = min(len(energies), j + smooth_radius + 1)
            smoothed.append(sum(energies[lo:hi]) / (hi - lo))

        sorted_e = sorted(smoothed)
        p5 = sorted_e[max(0, len(sorted_e) // 20)]
        p50 = sorted_e[len(sorted_e) // 2]
        threshold = min(p5 * 4.0 + 10, (p5 + p50) / 2)

        stretches = []
        in_silence = False
        s_start = 0
        for j, e in enumerate(smoothed):
            if e <= threshold:
                if not in_silence:
                    s_start = j
                    in_silence = True
            else:
                if in_silence:
                    dur_ms = (j - s_start) * frame_ms
                    if dur_ms >= MIN_SILENCE_MS:
                        stretches.append((s_start, j - 1))
                    in_silence = False
        if in_silence:
            dur_ms = (len(smoothed) - s_start) * frame_ms
            if dur_ms >= MIN_SILENCE_MS:
                stretches.append((s_start, len(smoothed) - 1))

        return stretches, energies, smoothed

    for i in range(1, len(pages)):
        prev_end = pages[i - 1].get("end_ms") or 0
        curr_start = pages[i].get("start_ms") or prev_end
        gap = curr_start - prev_end
        nominal_boundary_ms = (prev_end + curr_start) // 2

        if gap <= 50:
            margins = [2000, 3500, 5000]
        elif gap <= 500:
            margins = [1000, 2000, 3000]
        else:
            margins = [500, 1000, 2000]

        best_stretch = None
        best_search_lo_ms = 0

        for margin_ms in margins:
            search_lo_ms = max(0, prev_end - margin_ms)
            search_hi_ms = min(audio_duration_ms, curr_start + margin_ms)
            stretches, _, _ = _find_silences_in_window(search_lo_ms, search_hi_ms)

            if stretches:
                nominal_frame = max(0, (nominal_boundary_ms - search_lo_ms) // frame_ms)

                def score(stretch):
                    dur = (stretch[1] - stretch[0] + 1) * frame_ms
                    mid = (stretch[0] + stretch[1]) // 2
                    dist_ms = abs(mid - nominal_frame) * frame_ms
                    return dur * 1.5 - dist_ms

                best_stretch = max(stretches, key=score)
                best_search_lo_ms = search_lo_ms
                break

        if best_stretch is None:
            search_lo_ms = max(0, prev_end - margins[-1])
            search_hi_ms = min(audio_duration_ms, curr_start + margins[-1])
            _, _, smoothed = _find_silences_in_window(search_lo_ms, search_hi_ms)
            if smoothed:
                min_idx = min(range(len(smoothed)), key=lambda j: smoothed[j])
                split_points[i] = search_lo_ms + min_idx * frame_ms
            pg_prev = pages[i - 1].get("page", "?")
            pg_curr = pages[i].get("page", "?")
            print(f"    Page {pg_prev}->{pg_curr}: gap={gap}ms, "
                  f"NO silence found, fallback to min energy @{split_points[i]}ms")
            continue

        silence_start_ms = best_search_lo_ms + best_stretch[0] * frame_ms
        silence_end_ms = best_search_lo_ms + (best_stretch[1] + 1) * frame_ms
        silence_dur = silence_end_ms - silence_start_ms

        lead_in = min(int(silence_dur * 0.15), 50)
        split_ms = silence_end_ms - lead_in
        split_points[i] = split_ms

        pg_prev = pages[i - 1].get("page", "?")
        pg_curr = pages[i].get("page", "?")
        print(f"    Page {pg_prev}->{pg_curr}: gap={gap}ms, "
              f"silence={silence_start_ms}-{silence_end_ms}ms ({silence_dur}ms), "
              f"split@{split_ms}ms")

    for i in range(1, len(split_points)):
        split_points[i] = max(split_points[i], split_points[i - 1])
        split_points[i] = min(split_points[i], audio_duration_ms)

    return split_points


def _refine_verse_splits_with_energy(
    audio_path: str,
    segments: list[dict],
    split_points: list[int],
    audio_duration_ms: int,
) -> list[int]:
    """
    Refine verse split points where alignment gap < 100ms using energy analysis.
    Finds the lowest-energy point within ±500ms of each tight boundary.
    """
    needs_refinement = []
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].get("end_ms") or 0
        curr_start = segments[i].get("start_ms") or prev_end
        if curr_start - prev_end < 100:
            needs_refinement.append(i)

    if not needs_refinement:
        return split_points

    print(f"  Analyzing audio energy for {len(needs_refinement)} verse boundaries...")

    samples, sample_rate = _load_raw_pcm(audio_path)
    frame_ms = 10
    frame_samples = int(frame_ms * sample_rate / 1000)
    window_ms = 500

    for idx in needs_refinement:
        prev_end = segments[idx - 1].get("end_ms") or 0
        curr_start = segments[idx].get("start_ms") or prev_end
        boundary_ms = (prev_end + curr_start) // 2

        prev_seg_start = segments[idx - 1].get("start_ms") or 0
        next_seg_end = segments[idx].get("end_ms") or audio_duration_ms
        search_lo_ms = max(prev_seg_start, boundary_ms - window_ms)
        search_hi_ms = min(next_seg_end, boundary_ms + window_ms)

        search_lo = int(search_lo_ms * sample_rate / 1000)
        search_hi = min(len(samples), int(search_hi_ms * sample_rate / 1000))

        min_energy = float("inf")
        min_pos = int(boundary_ms * sample_rate / 1000)

        for pos in range(search_lo, search_hi - frame_samples, frame_samples):
            frame = samples[pos:pos + frame_samples]
            energy = sum(s * s for s in frame)
            if energy < min_energy:
                min_energy = energy
                min_pos = pos + frame_samples // 2

        dip_ms = int(min_pos * 1000 / sample_rate)
        split_points[idx] = dip_ms + 30

    for i in range(1, len(split_points)):
        split_points[i] = max(split_points[i], split_points[i - 1])
        split_points[i] = min(split_points[i], audio_duration_ms)

    return split_points


# ---------------------------------------------------------------------------
# Transcription with caching (reciter-scoped)
# ---------------------------------------------------------------------------

def get_alignment_data(
    surah_num: int,
    verses: list[dict],
    input_file: Path,
    timestamps_dir: Path,
    model_size: str,
    force_retranscribe: bool,
) -> dict | None:
    """Transcribe and align (or load cached), return alignment dict."""
    timestamps_file = timestamps_dir / f"{surah_num:03d}.json"

    alignment_data = None
    if timestamps_file.exists() and not force_retranscribe:
        print(f"  Loading cached timestamps from {timestamps_file}")
        with open(timestamps_file, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)
        if "verses" not in alignment_data or len(alignment_data["verses"]) != len(verses):
            print(f"  Cached timestamps invalid, re-transcribing...")
            alignment_data = None

    if alignment_data is None:
        recognized_words = transcribe_surah(str(input_file), model_size=model_size)
        if not recognized_words:
            print(f"  ERROR: No words recognised in surah {surah_num}")
            return None

        print(f"  Aligning {len(recognized_words)} words to reference text...")
        alignment_data = align_surah(recognized_words, verses)

        timestamps_dir.mkdir(parents=True, exist_ok=True)
        with open(timestamps_file, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved timestamps to {timestamps_file}")

    return alignment_data


# ---------------------------------------------------------------------------
# Page marker generation
# ---------------------------------------------------------------------------

def generate_page_markers(
    surah_num: int,
    verses: list[dict],
    reciter: str,
    model_size: str = "base",
    force_retranscribe: bool = False,
) -> bool:
    """Generate page boundary markers for a single surah."""
    input_file = RECITATIONS_DIR / reciter / f"{surah_num:03d}.mp3"
    timestamps_dir = TIMESTAMPS_DIR / reciter
    markers_dir = MARKERS_DIR / reciter / "pages"

    if not input_file.exists():
        print(f"  WARNING: {input_file} not found, skipping")
        return False

    alignment_data = get_alignment_data(
        surah_num, verses, input_file, timestamps_dir,
        model_size, force_retranscribe,
    )
    if alignment_data is None:
        return False

    audio_duration_ms = get_audio_duration_ms(str(input_file))
    if audio_duration_ms is None:
        print(f"  ERROR: Could not determine audio duration")
        return False

    pages = alignment_data["pages_madani"]
    if len(pages) <= 1:
        markers_dir.mkdir(parents=True, exist_ok=True)
        markers_file = markers_dir / f"{surah_num:03d}.json"
        with open(markers_file, "w", encoding="utf-8") as f:
            json.dump({"markers": []}, f, indent=2)
        print(f"  Single page, wrote empty markers")
        return True

    split_points = _compute_split_points(pages, audio_duration_ms)
    split_points = _refine_page_splits_with_energy(
        str(input_file), pages, split_points, audio_duration_ms,
    )

    markers = [{"time": split_points[i] / 1000.0} for i in range(1, len(pages))]

    markers_dir.mkdir(parents=True, exist_ok=True)
    markers_file = markers_dir / f"{surah_num:03d}.json"
    with open(markers_file, "w", encoding="utf-8") as f:
        json.dump({"markers": markers}, f, indent=2)

    print(f"  Wrote {len(markers)} page markers -> {markers_file}")
    return True


# ---------------------------------------------------------------------------
# Verse marker generation
# ---------------------------------------------------------------------------

def generate_verse_markers(
    surah_num: int,
    verses: list[dict],
    reciter: str,
    model_size: str = "base",
    force_retranscribe: bool = False,
) -> bool:
    """Generate verse boundary markers for a single surah."""
    input_file = RECITATIONS_DIR / reciter / f"{surah_num:03d}.mp3"
    timestamps_dir = TIMESTAMPS_DIR / reciter
    markers_dir = MARKERS_DIR / reciter / "verses"

    if not input_file.exists():
        print(f"  WARNING: {input_file} not found, skipping")
        return False

    alignment_data = get_alignment_data(
        surah_num, verses, input_file, timestamps_dir,
        model_size, force_retranscribe,
    )
    if alignment_data is None:
        return False

    audio_duration_ms = get_audio_duration_ms(str(input_file))
    if audio_duration_ms is None:
        print(f"  ERROR: Could not determine audio duration")
        return False

    verses_ts = alignment_data["verses"]
    segments = []
    segment_labels = []

    # Bismillah / intro: audio before verse 1 (skip for surah 9 / At-Tawbah)
    first_start = verses_ts[0].get("start_ms")
    if first_start and first_start > 50 and surah_num != 9:
        segments.append({"start_ms": 0, "end_ms": first_start})
        segment_labels.append(0)

    for vt in verses_ts:
        segments.append({"start_ms": vt["start_ms"], "end_ms": vt["end_ms"]})
        segment_labels.append(vt["ayah"])

    split_points = _compute_split_points(segments, audio_duration_ms)
    split_points = _refine_verse_splits_with_energy(
        str(input_file), segments, split_points, audio_duration_ms,
    )

    markers = [
        {"verse": segment_labels[i], "time": split_points[i] / 1000.0}
        for i in range(len(segment_labels))
    ]

    markers_dir.mkdir(parents=True, exist_ok=True)
    markers_file = markers_dir / f"{surah_num:03d}.json"
    with open(markers_file, "w", encoding="utf-8") as f:
        json.dump({"markers": markers}, f, indent=2)

    print(f"  Wrote {len(markers)} verse markers -> {markers_file}")
    return True


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_int_set(arg: str) -> set[int]:
    """Parse '1-6,17,20,25,103-114' into a set of ints."""
    result: set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def discover_reciters() -> list[str]:
    """Auto-discover reciter folders inside recitations/."""
    if not RECITATIONS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in RECITATIONS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate page and/or verse boundary markers for Quran recitations."
    )
    parser.add_argument(
        "--reciter", default=None,
        help="Reciter folder name(s), comma-separated (default: all in recitations/)",
    )
    parser.add_argument(
        "--mode", choices=["pages", "verses", "both"], default="both",
        help="Marker mode (default: both)",
    )
    parser.add_argument(
        "--surah", default="all",
        help="Surah(s) to process: '17', '7-16', '1,5,10', or 'all'",
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--skip", default="",
        help="Surah(s) to skip: '1-6,17,20,25,103-114'",
    )
    parser.add_argument(
        "--force-retranscribe", action="store_true",
        help="Force re-transcription even if timestamps exist",
    )

    args = parser.parse_args()

    # Resolve reciters
    if args.reciter:
        reciters = [r.strip() for r in args.reciter.split(",")]
    else:
        reciters = discover_reciters()
        if not reciters:
            print("No reciter folders found in recitations/. "
                  "Add MP3s to recitations/<reciter_name>/001.mp3 … 114.mp3")
            sys.exit(1)

    # Validate reciter folders exist
    for r in reciters:
        rdir = RECITATIONS_DIR / r
        if not rdir.is_dir():
            print(f"ERROR: recitations/{r}/ does not exist", file=sys.stderr)
            sys.exit(1)

    # Build surah list
    if args.surah.lower() == "all":
        surahs = list(range(1, 115))
    else:
        surahs = sorted(parse_int_set(args.surah))

    skip = parse_int_set(args.skip) if args.skip else set()
    surahs = [s for s in surahs if s not in skip]

    print("Loading Quran data...")
    surah_verses = load_quran_data()

    do_pages = args.mode in ("pages", "both")
    do_verses = args.mode in ("verses", "both")

    total_ok = 0
    total_fail = 0

    for reciter in reciters:
        print(f"\n{'#'*60}")
        print(f"# Reciter: {reciter}")
        print(f"{'#'*60}")

        for idx, sn in enumerate(surahs, 1):
            if sn not in surah_verses:
                print(f"WARNING: Surah {sn} not in quran.json")
                total_fail += 1
                continue

            verses = surah_verses[sn]
            print(f"\n[{idx}/{len(surahs)}] Surah {sn} ({len(verses)} verses)")

            if do_pages:
                print(f"  --- Page markers ---")
                if generate_page_markers(sn, verses, reciter,
                                         model_size=args.model,
                                         force_retranscribe=args.force_retranscribe):
                    total_ok += 1
                else:
                    total_fail += 1

            if do_verses:
                print(f"  --- Verse markers ---")
                if generate_verse_markers(sn, verses, reciter,
                                          model_size=args.model,
                                          force_retranscribe=args.force_retranscribe):
                    total_ok += 1
                else:
                    total_fail += 1

    print(f"\n{'='*60}")
    print(f"DONE: {total_ok} succeeded, {total_fail} failed")
    print(f"{'='*60}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
