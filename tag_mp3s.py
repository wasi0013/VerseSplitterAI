#!/usr/bin/env python3
"""Tag surah-based Quran MP3 files with ID3 metadata.

Usage:
    python tag_mp3s.py <reciter_folder> [--reciter NAME] [--cover PATH] [--album NAME] [--dry-run]

Supports two filename patterns:
  - Surah-based:  001.mp3 … 114.mp3
  - Page-based:   page<NNN>-<SSS><PPP>.mp3  (page, surah, part)

Examples:
    python tag_mp3s.py recitations/alafasy --reciter "Mishary Rashid Alafasy" --cover resources/1ayah-logo.png
    python tag_mp3s.py docs/examples/page_by_page/alafasy --reciter "Mishary Rashid Alafasy" --cover resources/1ayah-logo.png
    python tag_mp3s.py recitations/luhaidan --reciter "Salman Al-Utaybi"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from mutagen.id3 import (
    APIC,
    ID3,
    TALB,
    TCOM,
    TCON,
    TIT2,
    TPE1,
    TPE2,
    TRCK,
)

SURAH_NAMES = {
    1: "Al-Fatihah", 2: "Al-Baqarah", 3: "Aal-E-Imran", 4: "An-Nisa",
    5: "Al-Ma'idah", 6: "Al-An'am", 7: "Al-A'raf", 8: "Al-Anfal",
    9: "At-Tawbah", 10: "Yunus", 11: "Hud", 12: "Yusuf",
    13: "Ar-Ra'd", 14: "Ibrahim", 15: "Al-Hijr", 16: "An-Nahl",
    17: "Al-Isra", 18: "Al-Kahf", 19: "Maryam", 20: "Ta-Ha",
    21: "Al-Anbiya", 22: "Al-Hajj", 23: "Al-Mu'minun", 24: "An-Nur",
    25: "Al-Furqan", 26: "Ash-Shu'ara", 27: "An-Naml", 28: "Al-Qasas",
    29: "Al-Ankabut", 30: "Ar-Rum", 31: "Luqman", 32: "As-Sajdah",
    33: "Al-Ahzab", 34: "Saba", 35: "Fatir", 36: "Ya-Sin",
    37: "As-Saffat", 38: "Sad", 39: "Az-Zumar", 40: "Ghafir",
    41: "Fussilat", 42: "Ash-Shura", 43: "Az-Zukhruf", 44: "Ad-Dukhan",
    45: "Al-Jathiyah", 46: "Al-Ahqaf", 47: "Muhammad", 48: "Al-Fath",
    49: "Al-Hujurat", 50: "Qaf", 51: "Adh-Dhariyat", 52: "At-Tur",
    53: "An-Najm", 54: "Al-Qamar", 55: "Ar-Rahman", 56: "Al-Waqi'ah",
    57: "Al-Hadid", 58: "Al-Mujadila", 59: "Al-Hashr", 60: "Al-Mumtahanah",
    61: "As-Saf", 62: "Al-Jumu'ah", 63: "Al-Munafiqun", 64: "At-Taghabun",
    65: "At-Talaq", 66: "At-Tahrim", 67: "Al-Mulk", 68: "Al-Qalam",
    69: "Al-Haqqah", 70: "Al-Ma'arij", 71: "Nuh", 72: "Al-Jinn",
    73: "Al-Muzzammil", 74: "Al-Muddaththir", 75: "Al-Qiyamah", 76: "Al-Insan",
    77: "Al-Mursalat", 78: "An-Naba", 79: "An-Nazi'at", 80: "Abasa",
    81: "At-Takwir", 82: "Al-Infitar", 83: "Al-Mutaffifin", 84: "Al-Inshiqaq",
    85: "Al-Buruj", 86: "At-Tariq", 87: "Al-A'la", 88: "Al-Ghashiyah",
    89: "Al-Fajr", 90: "Al-Balad", 91: "Ash-Shams", 92: "Al-Layl",
    93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin", 96: "Al-Alaq",
    97: "Al-Qadr", 98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat",
    101: "Al-Qari'ah", 102: "At-Takathur", 103: "Al-Asr", 104: "Al-Humazah",
    105: "Al-Fil", 106: "Quraysh", 107: "Al-Ma'un", 108: "Al-Kawthar",
    109: "Al-Kafirun", 110: "An-Nasr", 111: "Al-Masad", 112: "Al-Ikhlas",
    113: "Al-Falaq", 114: "An-Nas",
}

# Pattern: 001.mp3 … 114.mp3
FILENAME_RE = re.compile(r"^(\d{3})\.mp3$")

# Pattern: page001-001000.mp3 (page<page>-<surah><part>.mp3)
PAGE_FILENAME_RE = re.compile(r"^page(\d{3})-(\d{3})(\d{3})\.mp3$")


def load_quran_data(quran_json_path: Path) -> dict:
    """Load quran.json and return a dict keyed by chapter number."""
    with open(quran_json_path) as f:
        return json.load(f)


def parse_filename(filename: str) -> int | None:
    """Extract surah number from filename like 001.mp3."""
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return int(m.group(1))


def parse_page_filename(filename: str) -> tuple[int, int, int] | None:
    """Extract page, surah, and part from filename like page321-020009.mp3."""
    m = PAGE_FILENAME_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def tag_page_mp3s(
    mp3_folder: Path,
    reciter: str,
    cover_path: Path | None,
    album: str,
    quran_json_path: Path,
    dry_run: bool = False,
) -> None:
    quran_data = load_quran_data(quran_json_path)

    cover_data = None
    mime = "image/png"
    if cover_path and cover_path.exists():
        mime = "image/png" if cover_path.suffix.lower() == ".png" else "image/jpeg"
        cover_data = cover_path.read_bytes()
        print(f"Using cover art: {cover_path} ({mime})")
    elif cover_path:
        print(f"Warning: cover file not found: {cover_path}", file=sys.stderr)

    mp3_files = sorted(mp3_folder.glob("page*.mp3"))
    if not mp3_files:
        print(f"No page MP3 files found in {mp3_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mp3_files)} page MP3 files in {mp3_folder}")

    for mp3_file in mp3_files:
        parsed = parse_page_filename(mp3_file.name)
        if parsed is None:
            print(f"  Skipping (unrecognized filename): {mp3_file.name}")
            continue

        page, surah, part = parsed
        surah_name = SURAH_NAMES.get(surah, f"Surah {surah}")
        title = f"{page:03d} {surah:03d} {surah_name}"
        track_num = str(page)

        if dry_run:
            print(f"  [DRY RUN] {mp3_file.name} -> title={title!r}")
            continue

        try:
            tags = ID3(mp3_file)
        except Exception:
            tags = ID3()

        tags.delete(mp3_file)
        tags = ID3()

        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=reciter))
        tags.add(TPE2(encoding=3, text=reciter))
        tags.add(TALB(encoding=3, text=album))
        tags.add(TCON(encoding=3, text="Quran"))
        tags.add(TCOM(encoding=3, text=reciter))
        tags.add(TRCK(encoding=3, text=track_num))

        if cover_data:
            tags.add(
                APIC(
                    encoding=3,
                    mime=mime,
                    type=3,
                    desc="Cover",
                    data=cover_data,
                )
            )

        tags.save(mp3_file)
        print(f"  Tagged: {mp3_file.name} -> {title}")

    print("Done!")


def tag_mp3s(
    mp3_folder: Path,
    reciter: str,
    cover_path: Path | None,
    album: str,
    dry_run: bool = False,
) -> None:
    cover_data = None
    mime = "image/png"
    if cover_path and cover_path.exists():
        mime = "image/png" if cover_path.suffix.lower() == ".png" else "image/jpeg"
        cover_data = cover_path.read_bytes()
        print(f"Using cover art: {cover_path} ({mime})")
    elif cover_path:
        print(f"Warning: cover file not found: {cover_path}", file=sys.stderr)

    mp3_files = sorted(mp3_folder.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {mp3_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mp3_files)} MP3 files in {mp3_folder}")

    for mp3_file in mp3_files:
        surah = parse_filename(mp3_file.name)
        if surah is None:
            print(f"  Skipping (unrecognized filename): {mp3_file.name}")
            continue

        surah_name = SURAH_NAMES.get(surah, f"Surah {surah}")
        title = f"{surah:03d} {surah_name}"
        track_num = str(surah)

        if dry_run:
            print(f"  [DRY RUN] {mp3_file.name} -> title={title!r}")
            continue

        try:
            tags = ID3(mp3_file)
        except Exception:
            tags = ID3()

        tags.delete(mp3_file)
        tags = ID3()

        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=reciter))
        tags.add(TPE2(encoding=3, text=reciter))
        tags.add(TALB(encoding=3, text=album))
        tags.add(TCON(encoding=3, text="Quran"))
        tags.add(TCOM(encoding=3, text=reciter))
        tags.add(TRCK(encoding=3, text=track_num))

        if cover_data:
            tags.add(
                APIC(
                    encoding=3,
                    mime=mime,
                    type=3,
                    desc="Cover",
                    data=cover_data,
                )
            )

        tags.save(mp3_file)
        print(f"  Tagged: {mp3_file.name} -> {title}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Tag surah-based Quran MP3 files with ID3 metadata."
    )
    parser.add_argument("mp3_folder", type=Path, help="Folder containing surah MP3s (e.g. recitations/alafasy)")
    parser.add_argument(
        "--reciter", default="Unknown Reciter",
        help="Reciter name for artist metadata",
    )
    parser.add_argument(
        "--cover", type=Path, default=None, help="Path to album cover image (PNG/JPG)",
    )
    parser.add_argument(
        "--album", default=None, help="Album name (default: 'Quran - <reciter>')",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--quran-json", type=Path, default=Path("quran.json"),
        help="Path to quran.json for page-based tagging (default: quran.json)",
    )

    args = parser.parse_args()

    if not args.mp3_folder.is_dir():
        print(f"Error: {args.mp3_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    album = args.album or f"Quran - {args.reciter}"

    # Detect mode: page-based if any page*.mp3 files exist, else surah-based
    has_page_files = any(args.mp3_folder.glob("page*.mp3"))
    has_surah_files = any(FILENAME_RE.match(f.name) for f in args.mp3_folder.glob("*.mp3"))

    if has_page_files:
        tag_page_mp3s(
            mp3_folder=args.mp3_folder,
            reciter=args.reciter,
            cover_path=args.cover,
            album=album,
            quran_json_path=args.quran_json,
            dry_run=args.dry_run,
        )
    elif has_surah_files:
        tag_mp3s(
            mp3_folder=args.mp3_folder,
            reciter=args.reciter,
            cover_path=args.cover,
            album=album,
            dry_run=args.dry_run,
        )
    else:
        print(
            f"Error: no recognized MP3 files in {args.mp3_folder}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
