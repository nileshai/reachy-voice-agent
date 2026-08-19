"""Teacher module — read material aloud, take questions, carry on.

The robot's audio API has no seek, no pause and no position query: only
upload, play_sound and stop_sound. So "pause" is not a thing that can be
asked for -- material is split into short segments, played in sequence, and
the position is tracked here. Resuming means replaying segment N, which is
also why the interrupted segment is repeated rather than skipped: you never
lose the sentence you talked over.

Knowing the segment index is what makes questions answerable. When you
interrupt, the text you were hearing is `segments[pos]`, and its neighbours
are the context the model needs.

Phase 0: hardcoded text, proving play -> interrupt -> answer -> resume.
Ingestion (web, files, YouTube) comes next and only has to produce a list of
strings, so nothing below changes.
"""

from __future__ import annotations

import re
import threading
import time

# Roughly 15 s of speech. Short enough that an interruption costs little to
# replay, long enough that the gap between segments stays infrequent.
SEG_TARGET_CHARS = 220
SEG_MAX_CHARS = 320

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def segment(text: str) -> list[str]:
    """Split prose into speakable segments on sentence boundaries.

    Sentence boundaries matter: resuming mid-clause sounds broken, and the
    interrupted segment is replayed in full.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    out: list[str] = []
    cur = ""
    for sent in _SENTENCE.split(text):
        if not sent:
            continue
        # A single sentence longer than the max gets broken on commas rather
        # than left as a 40-second monologue no one can interrupt cleanly.
        pieces = [sent]
        while len(pieces[-1]) > SEG_MAX_CHARS:
            tail = pieces.pop()
            cut = tail.rfind(",", 0, SEG_MAX_CHARS)
            cut = cut + 1 if cut > SEG_TARGET_CHARS // 2 else SEG_MAX_CHARS
            pieces += [tail[:cut].strip(), tail[cut:].strip()]
        for piece in pieces:
            if cur and len(cur) + len(piece) + 1 > SEG_TARGET_CHARS:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        out.append(cur)
    return out


class Teacher:
    """Plays material segment by segment, pausing on demand."""

    def __init__(self, speech, play, stop_sound, emit, set_state):
        self.material = None             # content.Material when one is loaded
        self.speech = speech
        self._play = play                # play_wav(bytes, name) -> duration
        self._stop_sound = stop_sound
        self.emit = emit
        self._set_state = set_state

        self.title = ""
        self.segments: list[str] = []
        self.audio: list = []            # parallel wav chunks, for audio mode
        self.pos = 0
        self.active = False
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._nth = 0

    # -- lifecycle ------------------------------------------------------
    def load(self, text: str, title: str = "") -> int:
        self.stop()
        self.material = None
        self.audio = []
        self.segments = segment(text)
        self.title = title
        self.pos = 0
        return len(self.segments)

    def load_material(self, material) -> int:
        """Play prepared material: text read aloud, or original audio chunks."""
        self.stop()
        self.material = material
        self.title = material.title
        self.segments = material.segments
        self.audio = list(material.audio)
        self.pos = 0
        return len(material)

    def start(self) -> bool:
        if not self.segments or self.active:
            return False
        self.active = True
        self._paused.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.emit("lesson_start", title=self.title, segments=len(self.segments))
        return True

    def pause(self) -> None:
        """Stop mid-segment. The position is deliberately not advanced."""
        if not self.active or self._paused.is_set():
            return
        self._paused.set()
        self._stop_sound()
        self.emit("lesson_pause", at=self.pos)

    def resume(self) -> None:
        if not self.active or not self._paused.is_set():
            return
        self.emit("lesson_resume", at=self.pos)
        self._paused.clear()

    def stop(self) -> None:
        self.active = False
        self._paused.set()
        try:
            self._stop_sound()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    # -- what the model needs to answer a question ----------------------
    def context(self, window: int = 1) -> str:
        """The text around where the listener interrupted."""
        if not self.segments:
            return self._transcribe_here()
        lo = max(0, self.pos - window)
        hi = min(len(self.segments), self.pos + window + 1)
        text = " ".join(s for s in self.segments[lo:hi] if s.strip())
        return text or self._transcribe_here()

    def _transcribe_here(self) -> str:
        """Transcribe just the chunk being played, for videos with no captions.

        Only the current chunk, and only when a question is actually asked --
        transcribing an entire video up front would cost far more and most of
        it would never be needed.
        """
        if not self.audio or self.pos >= len(self.audio):
            return ""
        try:
            import wave as _wave, io as _io
            import numpy as _np
            with _wave.open(str(self.audio[self.pos])) as w:
                pcm = w.readframes(w.getnframes())
                rate = w.getframerate()
            a = _np.frombuffer(pcm, dtype="<i2")
            if rate != 16000:                      # Riva wants 16 kHz
                idx = (_np.arange(int(len(a) * 16000 / rate)) * rate // 16000)
                a = a[_np.clip(idx, 0, len(a) - 1)]
            text = self.speech.transcribe(a.astype("<i2").tobytes())
            if text:
                self.emit("lesson_transcribed", n=self.pos + 1, text=text[:70])
                # Cache it so a second question about the same chunk is free.
                while len(self.segments) <= self.pos:
                    self.segments.append("")
                self.segments[self.pos] = text
            return text
        except Exception as e:
            self.emit("error", where="transcribe", detail=f"{type(e).__name__}: {e}")
            return ""

    # -- listener commands ----------------------------------------------
    def again(self) -> None:
        """Replay the current segment from the start."""
        self.pos = max(0, self.pos)
        self._restart_segment()

    def back(self, n: int = 1) -> None:
        self.pos = max(0, self.pos - n)
        self._restart_segment()

    def skip(self, n: int = 1) -> None:
        self.pos = min(self._count(), self.pos + n)
        self._restart_segment()

    def _restart_segment(self) -> None:
        self.emit("lesson_seek", to=self.pos + 1, of=self._count())
        if self.active:
            self._stop_sound()          # cut current audio; loop replays pos
            self._paused.clear()

    def so_far(self) -> str:
        """Everything heard up to now, for a summary request."""
        return " ".join(s for s in self.segments[:self.pos + 1] if s.strip())

    def _count(self) -> int:
        return len(self.audio) if self.audio else len(self.segments)

    @property
    def progress(self) -> str:
        if not self._count():
            return ""
        return f"{min(self.pos + 1, self._count())}/{self._count()}"

    # -- playback -------------------------------------------------------
    def _loop(self) -> None:
        while self.active and self.pos < self._count():
            if self._paused.is_set():
                time.sleep(0.15)
                continue
            seg = self.segments[self.pos] if self.pos < len(self.segments) else ""
            try:
                if self.audio:
                    # Original audio: play the chunk as-is. Nothing is
                    # synthesised, so the speaker's own voice is preserved.
                    wav = self.audio[self.pos].read_bytes()
                else:
                    wav = self.speech.synthesize(seg)
                if not wav:
                    self.pos += 1
                    continue
                # Alternate names: uploading over the file being played
                # interrupts it on the robot.
                self._nth += 1
                self._set_state("speaking")
                dur = self._play(wav, f"_reachy_lesson_{self._nth % 2}.wav")
            except Exception as e:
                self.emit("error", where="lesson", detail=f"{type(e).__name__}: {e}")
                time.sleep(1.0)
                continue
            self.emit("lesson_segment", n=self.pos + 1,
                      of=self._count(), text=(seg[:70] or "(audio)"))
            # Wait out the audio, but wake immediately on pause. Returning
            # True means we were interrupted, so pos is left alone and the
            # segment is replayed on resume.
            if self._paused.wait(dur + 0.1):
                continue
            self.pos += 1
        if self.active and self.pos >= self._count():
            self.emit("lesson_done", title=self.title)
            self.active = False
            self._set_state("idle")
