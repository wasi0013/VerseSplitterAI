
# VerseSplitterAI

VerseSplitterAI is a standalone tool for generating Quran recitation markers at two levels:

- Page boundaries for page-by-page playback
- Verse boundaries for verse-by-verse playback

It is designed for batch processing multiple reciters from a consistent folder layout and produces reusable JSON marker files for the [Audio Splitter project](https://github.com/wasi0013/audio-splitter/tree/cloudflare)
. So that the slight inconsistencies from AI can be manually adjusted as needed before publishing to the  downstream apps and publishing workflows such as [Murajah](https://murajah.pages.dev/) or [1Ayah](https://1ayah.pages.dev)

![1Ayah logo](resources/1ayah-logo.png)

Website: [1ayah.pages.dev](https://1ayah.pages.dev)

Android app: [Google Play](https://play.google.com/store/apps/details?id=com.oneayah.oneayah)

## Features

- Standalone `uv`-managed Python project
- Batch processing for multiple reciters
- Word-level transcription and alignment using Whisper
- Page marker generation based on Madani page boundaries
- Verse marker generation with bismillah handling
- Per-reciter timestamp caching to avoid unnecessary retranscription
- Optional MP3 metadata tagging for surah-based source files

## Project Layout

```text
recitations/
  alafasy/
    001.mp3
    002.mp3
    ...
  luhaidan/
    001.mp3
    002.mp3
    ...

timestamps/
  <reciter>/
    001.json
    ...

markers/
  <reciter>/
    pages/
      001.json
      ...
    verses/
      001.json
      ...
```

Input audio is expected in surah form:

- `recitations/<reciter>/001.mp3` through `114.mp3`

## Setup

```bash
uv sync
```

The project depends on `ffmpeg` and `ffprobe` being available on the system.

## Quick Start

For a first local run:

1. Create a reciter folder under `recitations/`, for example `recitations/alafasy/`.
2. Add surah MP3 files named `001.mp3` through `114.mp3`.
3. Install dependencies with `uv sync`.
4. Generate markers with `uv run python marker_generator.py --reciter alafasy`.

If you only want one output type while testing:

```bash
uv run python marker_generator.py --reciter alafasy --mode pages
uv run python marker_generator.py --reciter alafasy --mode verses
```

## Usage

Generate both page and verse markers for all reciters found in `recitations/`:

```bash
uv run python marker_generator.py
```

Generate markers for one reciter:

```bash
uv run python marker_generator.py --reciter alafasy
```

Process specific surahs only:

```bash
uv run python marker_generator.py --reciter alafasy --surah 78-114
```

Generate only page markers or only verse markers:

```bash
uv run python marker_generator.py --reciter alafasy --mode pages
uv run python marker_generator.py --reciter alafasy --mode verses
```

Force retranscription instead of using cached timestamps:

```bash
uv run python marker_generator.py --reciter alafasy --force-retranscribe
```

## Output

Page marker files are written to:

- `markers/<reciter>/pages/<surah>.json`

Verse marker files are written to:

- `markers/<reciter>/verses/<surah>.json`

Page marker format:

```json
{
  "markers": [
    { "time": 159.09 }
  ]
}
```

Verse marker format:

```json
{
  "markers": [
    { "verse": 0, "time": 0.0 },
    { "verse": 1, "time": 5.835 }
  ]
}
```

## MP3 Metadata Tagging

Use `tag_mp3s.py` to apply consistent metadata to source surah files:

```bash
uv run python tag_mp3s.py recitations/alafasy --reciter "Mishary Rashid Alafasy" --cover resources/1ayah-logo.png
```

This writes title, album, artist, track number, genre, and cover art metadata to the MP3s so that the audio players like vlc shows meaningful information along with the audio.

## Examples

Bundled page-by-page example audio is available under `docs/examples/page_by_page/alafasy/`.
These files are useful as a reference for page-level output naming and downstream integration.
