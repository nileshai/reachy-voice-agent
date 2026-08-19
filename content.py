"""Turn a source into something the robot can read aloud.

Every source reduces to the same shape: a title, a list of speakable text
segments, and optionally a list of audio chunk files. Text-only sources
(articles, files, pasted text) produce segments alone and are spoken by the
robot. YouTube produces both -- the original audio is played, while the
transcript rides alongside purely so questions can be answered about what was
just heard.

The audio path exists because the robot's playback API has no seek: a long
file cannot be paused and resumed, so anything played has to arrive as short
chunks that can be stopped between.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from teacher import segment

YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([\w-]{11})")

# Chunk length for original audio. Short enough to stop promptly on a
# question, long enough that the gap between chunks is infrequent.
AUDIO_CHUNK_S = 15


@dataclass
class Material:
    title: str = ""
    kind: str = "text"                    # text | audio
    segments: list[str] = field(default_factory=list)
    audio: list[Path] = field(default_factory=list)   # parallel to segments
    # For audio without captions: where each chunk starts, so a chunk can be
    # transcribed on demand rather than transcribing the whole video up front.
    starts: list[float] = field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.audio) if self.kind == "audio" else len(self.segments)


# ----------------------------------------------------------------- text

def from_text(text: str, title: str = "pasted text") -> Material:
    return Material(title=title, segments=segment(text), source="text")


def from_url(url: str) -> Material:
    """Extract the readable article from a web page."""
    html = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=30).read().decode("utf-8", "replace")
    text, title = "", url
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=False) or ""
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass
    if not text.strip():
        # Fallback so a missing library or an odd page still reads.
        text = re.sub(r"(?is)<(script|style|nav|footer|header).*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        m = re.search(r"(?is)<title>(.*?)</title>", html)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
    return Material(title=title.strip()[:120], segments=segment(text), source=url)


def from_file(path: str) -> Material:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".pdf":
        # pdftotext ships with poppler and is already present; -layout off
        # gives cleaner prose for reading aloud than preserving columns.
        out = subprocess.run(["pdftotext", "-q", str(p), "-"],
                             capture_output=True, text=True, timeout=120)
        text = out.stdout
    else:
        text = p.read_text(errors="replace")
    # Strip page numbers and hard-wrap artefacts that read badly aloud.
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return Material(title=p.name, segments=segment(text), source=str(p))


# -------------------------------------------------------------- youtube

def youtube_id(url: str) -> str | None:
    m = YOUTUBE_RE.search(url or "")
    return m.group(1) if m else None


def _captions(vid: str) -> list[dict]:
    """Timestamped captions, or [] if the video has none."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        try:                                   # newer API
            fetched = api.fetch(vid)
            return [{"start": s.start, "duration": s.duration, "text": s.text}
                    for s in fetched]
        except AttributeError:                 # older API
            return YouTubeTranscriptApi.get_transcript(vid)
    except Exception:
        return []


def from_youtube(url: str, workdir: Path, max_minutes: float = 30.0) -> Material:
    """Download the audio, cut it into chunks, and align captions to them."""
    vid = youtube_id(url)
    if not vid:
        raise ValueError("not a YouTube URL")
    workdir.mkdir(parents=True, exist_ok=True)

    import yt_dlp
    # YouTube 403s the default web client from most networks. The android
    # client still serves; the rest are fallbacks for when it stops.
    info = None
    last = None
    for client in (["android"], ["ios"], ["tv"], ["web_safari"]):
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(workdir / "src.%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True,
            "extractor_args": {"youtube": {"player_client": client}},
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as e:          # try the next client
            last = e
    if info is None:
        raise RuntimeError(f"could not download audio: {str(last)[:200]}")
    title = (info.get("title") or vid)[:120]
    src = next(iter(sorted(workdir.glob("src.*"))), None)
    if src is None:
        raise RuntimeError("download produced no file")

    # 22.05 kHz mono wav: what the robot wants, and a quarter the bytes of
    # 44.1 stereo for an upload that happens once per chunk.
    dur = min(float(info.get("duration") or 0) or 1e9, max_minutes * 60)
    pat = str(workdir / "chunk_%04d.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
         "-t", str(dur), "-ac", "1", "-ar", "22050",
         "-f", "segment", "-segment_time", str(AUDIO_CHUNK_S), pat],
        check=True, timeout=900)
    chunks = sorted(workdir.glob("chunk_*.wav"))
    starts = [i * AUDIO_CHUNK_S for i in range(len(chunks))]

    # Caption text per chunk, so a question at 4:32 is answered against what
    # was being said at 4:32.
    caps = _captions(vid)
    segs: list[str] = []
    for s in starts:
        window = [c["text"] for c in caps
                  if s <= c["start"] < s + AUDIO_CHUNK_S]
        segs.append(" ".join(window).strip())
    return Material(title=title, kind="audio", segments=segs, audio=chunks,
                    starts=starts, source=url)


# ------------------------------------------------------------ dispatch

def load(source: str, workdir: Path, minutes: float = 30.0) -> Material:
    """Whatever the user typed -> Material."""
    src = (source or "").strip()
    if not src:
        raise ValueError("nothing given")
    if youtube_id(src):
        return from_youtube(src, workdir, max_minutes=minutes)
    if src.startswith(("http://", "https://")):
        return from_url(src)
    p = Path(src).expanduser()
    if p.exists():
        return from_file(str(p))
    return from_text(src)
