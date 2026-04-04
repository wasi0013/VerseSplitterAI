"""
Quran Word-Level Aligner

Uses Whisper to transcribe surah audio with word-level timestamps,
then aligns transcribed words to known Quran text using DTW alignment.

Based on the approach from:
https://github.com/yazanbaker94/reciters-timing-data-V2

Adapted for full-surah audio files (vs. per-ayah files in the reference).
"""

import re
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import Levenshtein


@dataclass
class WordSegment:
    """A recognized word with timestamps."""
    word: str
    start_ms: int
    end_ms: int


@dataclass
class AlignedSpan:
    """A reference word index mapped to timestamps."""
    index_start: int
    index_end: int
    start_ms: int
    end_ms: int


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison (remove diacritics, normalize letters)."""
    # Remove diacritics (tashkeel)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize alef maqsura
    text = re.sub(r'ى', 'ي', text)
    # Remove tatweel (kashida)
    text = text.replace('\u0640', '')
    # Remove Quran-specific marks (sajda, rub el hizb, etc.)
    text = re.compile(r'[\u06D6-\u06ED]').sub('', text)
    # Remove zero-width characters
    text = re.compile(r'[\u200B-\u200F\u202A-\u202E\uFEFF]').sub('', text)
    # Small/superscript alef
    text = text.replace('\u0670', '')
    text = text.replace('\u06E5', '')
    text = text.replace('\u06E6', '')
    return text.strip()


def align_words(recognized: list[WordSegment], reference_words: list[str]) -> list[AlignedSpan]:
    """
    DTW alignment between Whisper-recognized words and reference Quran text.
    Returns aligned spans mapping reference word indices to timestamps.
    """
    if not recognized or not reference_words:
        return []

    rec_normalized = [normalize_arabic(w.word) for w in recognized]
    ref_normalized = [normalize_arabic(w) for w in reference_words]

    n, m = len(rec_normalized), len(ref_normalized)
    INF = float('inf')

    # DP table for DTW
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for j in range(1, m + 1):
        dp[0][j] = j * 0.8  # Skip reference word penalty
    for i in range(1, n + 1):
        dp[i][0] = i * 1.2  # Skip recognized word penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            rec_word = rec_normalized[i - 1]
            ref_word = ref_normalized[j - 1]

            if rec_word == ref_word:
                cost = 0
            else:
                ratio = Levenshtein.ratio(rec_word, ref_word)
                if ratio > 0.8:
                    cost = 0.1
                elif ratio > 0.6:
                    cost = 0.4
                elif ratio > 0.4:
                    cost = 0.7
                else:
                    cost = 1.0

            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,   # match/substitute
                dp[i - 1][j] + 1.2,         # skip recognized word
                dp[i][j - 1] + 0.8          # skip reference word
            )

    # Backtrace
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            rec_word = rec_normalized[i - 1]
            ref_word = ref_normalized[j - 1]

            if rec_word == ref_word:
                cost = 0
            else:
                ratio = Levenshtein.ratio(rec_word, ref_word)
                if ratio > 0.8:
                    cost = 0.1
                elif ratio > 0.6:
                    cost = 0.4
                elif ratio > 0.4:
                    cost = 0.7
                else:
                    cost = 1.0

            if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 0.001:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.2)) < 0.001:
            alignment.append((i - 1, None))
            i -= 1
        elif j > 0:
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()

    # Build spans from alignment
    spans = []
    for rec_idx, aligned_ref_idx in alignment:
        if rec_idx is not None and aligned_ref_idx is not None:
            spans.append(AlignedSpan(
                index_start=aligned_ref_idx,
                index_end=aligned_ref_idx + 1,
                start_ms=recognized[rec_idx].start_ms,
                end_ms=recognized[rec_idx].end_ms
            ))

    # Fill gaps — interpolate timestamps for unmatched reference words
    if spans and len(spans) < len(reference_words):
        covered_indices = {s.index_start for s in spans}
        span_by_idx = {s.index_start: s for s in spans}
        filled_spans = []

        for ref_idx in range(len(reference_words)):
            if ref_idx in covered_indices:
                filled_spans.append(span_by_idx[ref_idx])
            else:
                # Find next covered span
                next_span_idx = None
                for idx in range(ref_idx + 1, len(reference_words)):
                    if idx in covered_indices:
                        next_span_idx = idx
                        break

                prev_span = filled_spans[-1] if filled_spans else None
                next_span = span_by_idx.get(next_span_idx) if next_span_idx else None

                if prev_span and next_span:
                    gap_start = prev_span.end_ms
                    gap_end = next_span.start_ms

                    if gap_end > gap_start:
                        gap_words = next_span_idx - ref_idx
                        word_duration = (gap_end - gap_start) // max(gap_words, 1)
                        filled_spans.append(AlignedSpan(
                            index_start=ref_idx,
                            index_end=ref_idx + 1,
                            start_ms=gap_start,
                            end_ms=gap_start + word_duration
                        ))
                    else:
                        # Overlap — distribute within the next span
                        words_in_segment = list(range(ref_idx, next_span_idx + 1))
                        next_duration = next_span.end_ms - next_span.start_ms
                        word_lengths = [len(normalize_arabic(reference_words[i])) for i in words_in_segment]
                        total_length = sum(word_lengths)

                        current_start = next_span.start_ms
                        for i, word_idx in enumerate(words_in_segment):
                            proportion = word_lengths[i] / total_length if total_length > 0 else 1 / len(words_in_segment)
                            wd = int(next_duration * proportion)

                            if word_idx == ref_idx:
                                filled_spans.append(AlignedSpan(
                                    index_start=word_idx,
                                    index_end=word_idx + 1,
                                    start_ms=current_start,
                                    end_ms=current_start + wd
                                ))
                            elif word_idx == next_span_idx:
                                span_by_idx[next_span_idx] = AlignedSpan(
                                    index_start=next_span_idx,
                                    index_end=next_span_idx + 1,
                                    start_ms=current_start,
                                    end_ms=next_span.end_ms
                                )
                            current_start += wd
                elif prev_span:
                    avg_duration = max(prev_span.end_ms - prev_span.start_ms, 300)
                    filled_spans.append(AlignedSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start_ms=prev_span.end_ms,
                        end_ms=prev_span.end_ms + avg_duration
                    ))
                elif next_span:
                    avg_duration = max(next_span.end_ms - next_span.start_ms, 300)
                    filled_spans.append(AlignedSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start_ms=max(0, next_span.start_ms - avg_duration),
                        end_ms=next_span.start_ms
                    ))

        spans = filled_spans

    return spans


_cached_model = None
_cached_model_size = None


def get_whisper_model(model_size: str = "large-v3"):
    """Get or create a cached Whisper model instance."""
    global _cached_model, _cached_model_size
    if _cached_model is None or _cached_model_size != model_size:
        from faster_whisper import WhisperModel
        print(f"  Loading Whisper model '{model_size}'...")
        _cached_model = WhisperModel(model_size, device="auto", compute_type="auto")
        _cached_model_size = model_size
    return _cached_model


def transcribe_surah(audio_path: str, model_size: str = "large-v3") -> list[WordSegment]:
    """
    Transcribe a full surah audio file using faster-whisper with word timestamps.
    Converts MP3 to WAV first to avoid decoder issues with some MP3 files.
    Returns list of WordSegments with timestamps.
    """
    model = get_whisper_model(model_size)

    # Convert to 16kHz mono WAV to avoid MP3 decoder issues in faster-whisper
    import tempfile, os
    wav_path = None
    transcode_path = audio_path
    if audio_path.lower().endswith(".mp3"):
        wav_path = tempfile.mktemp(suffix=".wav")
        print(f"  Converting to WAV for reliable decoding...")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            transcode_path = wav_path
        else:
            print(f"  WAV conversion failed, using MP3 directly")
            wav_path = None

    print(f"  Transcribing {audio_path}...")
    segments, info = model.transcribe(
        transcode_path,
        language="ar",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=200,
        ),
    )

    words = []
    for segment in segments:
        if segment.words:
            for word_info in segment.words:
                word = word_info.word.strip()
                if word:
                    words.append(WordSegment(
                        word=word,
                        start_ms=int(word_info.start * 1000),
                        end_ms=int(word_info.end * 1000),
                    ))

    # Clean up temp WAV file
    if wav_path and os.path.exists(wav_path):
        os.unlink(wav_path)

    print(f"  Recognized {len(words)} words")
    return words


def align_surah(
    recognized_words: list[WordSegment],
    verses: list[dict],
) -> dict:
    """
    Align recognized words from a full surah to known verse texts.

    For manageable sizes (< 4000 ref words), aligns the full surah at once.
    For larger surahs, uses chunked alignment by groups of verses.

    Args:
        recognized_words: WordSegments from Whisper transcription
        verses: List of verse dicts with 'verse', 'text', 'page', 'indopak_page'

    Returns:
        Dict with word-level, verse-level, and page-level timestamps
    """
    if not recognized_words or not verses:
        return {"words": [], "verses": [], "pages_madani": [], "pages_indopak": []}

    surah_num = verses[0]["chapter"]

    # Build flat list of all reference words with verse mapping
    all_ref_words = []
    verse_word_ranges = []  # (start_idx, end_idx) in all_ref_words for each verse
    for v in verses:
        words = v["text"].split()
        start = len(all_ref_words)
        all_ref_words.extend(words)
        end = len(all_ref_words)
        verse_word_ranges.append((start, end))

    total_ref_words = len(all_ref_words)
    total_rec_words = len(recognized_words)
    print(f"  Reference: {total_ref_words} words across {len(verses)} verses")
    print(f"  Recognized: {total_rec_words} words")

    # Choose strategy based on size
    MAX_FULL_ALIGNMENT_WORDS = 3000

    if total_ref_words <= MAX_FULL_ALIGNMENT_WORDS:
        print(f"  Using full-surah alignment...")
        spans = align_words(recognized_words, all_ref_words)
    else:
        print(f"  Using chunked alignment (surah too large for full DTW)...")
        spans = _chunked_align(recognized_words, all_ref_words, verses, verse_word_ranges)

    # Build word-level result
    word_timestamps = []
    span_by_idx = {s.index_start: s for s in spans}
    for idx, ref_word in enumerate(all_ref_words):
        if idx in span_by_idx:
            s = span_by_idx[idx]
            word_timestamps.append({
                "word": ref_word,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
            })
        else:
            word_timestamps.append({
                "word": ref_word,
                "start_ms": None,
                "end_ms": None,
            })

    # Build verse-level timestamps
    verse_timestamps = []
    for i, v in enumerate(verses):
        w_start, w_end = verse_word_ranges[i]
        verse_words = word_timestamps[w_start:w_end]

        starts = [w["start_ms"] for w in verse_words if w["start_ms"] is not None]
        ends = [w["end_ms"] for w in verse_words if w["end_ms"] is not None]

        verse_timestamps.append({
            "surah": surah_num,
            "ayah": v["verse"],
            "start_ms": min(starts) if starts else None,
            "end_ms": max(ends) if ends else None,
            "page_madani": v["page"],
            "page_indopak": v["indopak_page"],
            "segments": [
                [w_start + j, w_start + j + 1,
                 verse_words[j]["start_ms"],
                 verse_words[j]["end_ms"]]
                for j in range(w_end - w_start)
            ],
        })

    # Build page-level timestamps
    pages_madani = _build_page_timestamps(verse_timestamps, "page_madani")
    pages_indopak = _build_page_timestamps(verse_timestamps, "page_indopak")

    return {
        "surah": surah_num,
        "total_words": total_ref_words,
        "total_verses": len(verses),
        "words": word_timestamps,
        "verses": verse_timestamps,
        "pages_madani": pages_madani,
        "pages_indopak": pages_indopak,
    }


def _chunked_align(
    recognized_words: list[WordSegment],
    all_ref_words: list[str],
    verses: list[dict],
    verse_word_ranges: list[tuple],
) -> list[AlignedSpan]:
    """
    Chunked alignment for large surahs.
    Groups consecutive pages into chunks of ~2000 ref words each,
    then uses a sequential cursor to prevent overlap between chunks.
    """
    total_rec = len(recognized_words)
    total_ref = len(all_ref_words)
    ratio = total_rec / max(total_ref, 1)

    # Group verses by Madani page
    page_groups: dict[int, list[int]] = {}
    for vi, v in enumerate(verses):
        pg = v["page"]
        if pg not in page_groups:
            page_groups[pg] = []
        page_groups[pg].append(vi)

    sorted_pages = sorted(page_groups.keys())

    # Build multi-page chunks of ~2000 ref words each
    CHUNK_TARGET = 2000
    chunks = []
    current_pages = []
    current_ref_start = None
    current_ref_end = 0

    for pg in sorted_pages:
        vis = page_groups[pg]
        pg_ref_start = verse_word_ranges[vis[0]][0]
        pg_ref_end = verse_word_ranges[vis[-1]][1]

        if current_ref_start is None:
            current_ref_start = pg_ref_start

        current_pages.append(pg)
        current_ref_end = pg_ref_end

        chunk_words = current_ref_end - current_ref_start
        if chunk_words >= CHUNK_TARGET:
            chunks.append((list(current_pages), current_ref_start, current_ref_end))
            current_pages = []
            current_ref_start = None

    if current_pages:
        chunks.append((list(current_pages), current_ref_start, current_ref_end))

    # Align each chunk sequentially with a cursor
    all_spans = []
    rec_cursor = 0
    REC_MARGIN = 500

    for ci, (page_list, ref_start, ref_end) in enumerate(chunks):
        chunk_ref_words = all_ref_words[ref_start:ref_end]
        n_ref = len(chunk_ref_words)

        if ci == len(chunks) - 1:
            est_rec_end = total_rec
        else:
            est_rec_end = min(total_rec, int(ref_end * ratio) + REC_MARGIN)
            est_rec_end = max(est_rec_end, rec_cursor + int(n_ref * ratio) + 50)

        chunk_rec = recognized_words[rec_cursor:est_rec_end]
        pages_str = f"{page_list[0]}-{page_list[-1]}" if len(page_list) > 1 else str(page_list[0])
        print(f"    Chunk {ci+1}/{len(chunks)} (pages {pages_str}): {n_ref} ref, {len(chunk_rec)} rec words")

        chunk_spans = align_words(chunk_rec, chunk_ref_words)

        for s in chunk_spans:
            s.index_start += ref_start
            s.index_end += ref_start

        if chunk_spans and ci < len(chunks) - 1:
            sorted_spans = sorted(chunk_spans, key=lambda s: s.index_start)
            tail_spans = sorted_spans[-5:]
            pivot_time = sorted(s.end_ms for s in tail_spans)[len(tail_spans) // 2]

            new_cursor = est_rec_end
            for ri in range(rec_cursor, min(est_rec_end, total_rec)):
                if recognized_words[ri].start_ms >= pivot_time:
                    new_cursor = ri
                    break

            rec_cursor = new_cursor

        all_spans.extend(chunk_spans)

    return all_spans


def _build_page_timestamps(verse_timestamps: list[dict], page_key: str) -> list[dict]:
    """Group verse timestamps by page and compute page-level start/end."""
    page_groups: dict[int, list[dict]] = {}
    for vt in verse_timestamps:
        pg = vt[page_key]
        if pg not in page_groups:
            page_groups[pg] = []
        page_groups[pg].append(vt)

    pages = []
    for pg in sorted(page_groups.keys()):
        group = page_groups[pg]
        starts = [v["start_ms"] for v in group if v["start_ms"] is not None]
        ends = [v["end_ms"] for v in group if v["end_ms"] is not None]

        pages.append({
            "page": pg,
            "start_ms": min(starts) if starts else None,
            "end_ms": max(ends) if ends else None,
            "first_ayah": group[0]["ayah"],
            "last_ayah": group[-1]["ayah"],
            "ayah_count": len(group),
        })

    return pages
