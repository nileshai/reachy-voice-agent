"""Reachy Mini live voice + vision agent, powered by build.nvidia.com.

A wake-word conversational loop that hears through the robot's mics, sees
through its camera, thinks with an NVIDIA-hosted LLM, and answers through the
robot's speaker.

Pipeline
--------
    one gst-launch process  ──┬─▶ 16 kHz mono PCM  ─▶ VAD ─▶ Parakeet ASR ─┐
    (single WebRTC peer)      └─▶ JPEG @ 0.5 fps   ─▶ VLM caption ─┐       │
                                                                   ▼       ▼
                                                        rolling world state + text
                                                                   │
                                                        NVIDIA LLM ▼
                                                        Magpie TTS ─▶ robot speaker

Everything cloud-side runs on build.nvidia.com:
  * LLM + VLM  — https://integrate.api.nvidia.com/v1  (OpenAI-compatible REST)
  * ASR + TTS  — grpc.nvcf.nvidia.com:443             (Riva gRPC, function-id)

Why one gstreamer process: the robot advertises a single WebRTC producer, so
two consumers fight over it. Audio goes to stdout, video to rotating JPEGs.

Run standalone (no changes to server.py):
    export NVIDIA_API_KEY=nvapi-...
    .venv/bin/python voice_agent.py

Or mount into the existing app:
    from voice_agent import router as voice_router
    app.include_router(voice_router)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ------------------------------------------------------------------- config

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
ROBOT_IP = os.environ.get("REACHY_IP", "192.168.1.2")
ROBOT = f"http://{ROBOT_IP}:8000"
SIGNALLING = f"ws://{ROBOT_IP}:8443"

NVIDIA_REST = "https://integrate.api.nvidia.com/v1"
RIVA_URI = "grpc.nvcf.nvidia.com:443"

# NVCF function ids (verified ACTIVE against this account on 2026-08-11)
FN_ASR_PARAKEET = "1598d209-5e27-4d3c-8079-4751568b1081"  # ai-parakeet-ctc-1_1b-asr
FN_ASR_WHISPER = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"   # ai-whisper-large-v3
FN_TTS_MAGPIE = "877104f7-e885-42b9-8de8-f6e4c6303969"    # ai-magpie-tts-multilingual
FN_TTS_CHATTERBOX = "ddacc747-1269-4fab-bfd9-8f593dead106"  # ai-chatterbox-multilingual-tts

# llama-3.1-8b is the default because it is the fastest model tested that
# reliably emits tool calls. mistral-nemotron answers faster (0.41s vs 1.15s)
# but returns `tool_calls: []` and explains itself instead of acting, so the
# robot cannot be driven by voice with it. gpt-oss-120b nests the arguments
# wrongly. Set REACHY_LLM=mistralai/mistral-nemotron for chat-only, no control.
LLM_MODEL = os.environ.get("REACHY_LLM", "meta/llama-3.1-8b-instruct")
VLM_MODEL = os.environ.get("REACHY_VLM", "nvidia/nemotron-nano-12b-v2-vl")

TTS_FUNCTION = FN_TTS_MAGPIE
TTS_VOICE = "Magpie-Multilingual"
ASR_FUNCTION = FN_ASR_PARAKEET

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# Peak-normalise TTS to just under clipping. Gain is capped so a near-silent
# synthesis result cannot be amplified into a wall of noise.
TTS_PEAK_TARGET = float(os.environ.get("REACHY_TTS_PEAK", 0.97))
TTS_MAX_GAIN = float(os.environ.get("REACHY_TTS_MAX_GAIN", 8.0))

# External agent delegation (files, web, shell). OFF by default and
# deliberately so: with it on, anything said within earshot of the robot
# becomes a command executed on this machine, unauthenticated. Enable only
# when you are alone and understand what the agent CLI is permitted to do.
AGENT_ENABLED = os.environ.get("REACHY_AGENT_ENABLED", "") not in ("", "0", "false")
AGENT_CMD = shlex.split(os.environ.get("REACHY_AGENT_CMD", "claude -p"))
AGENT_TIMEOUT = float(os.environ.get("REACHY_AGENT_TIMEOUT", 120))
AGENT_MAX_CHARS = int(os.environ.get("REACHY_AGENT_MAX_CHARS", 700))

# Audio constants. Rates are fixed by the pipeline, not preferences.
RATE = 16000
FRAME_MS = 100
FRAME_SAMPLES = RATE * FRAME_MS // 1000       # 1600
FRAME_BYTES = FRAME_SAMPLES * 2

# VAD is adaptive: thresholds are multiples of the measured room noise floor,
# with absolute minimums as a backstop. Measured on this robot, per 100 ms
# frame: noise floor 220-350, speech frames 5000-16000. A x4 gate therefore
# sits ~1200 in this room -- far above noise, far below speech -- and keeps
# working in a quieter or noisier one.
VAD_ON_MULT = float(os.environ.get("REACHY_VAD_ON_MULT", 6.0))
VAD_OFF_MULT = float(os.environ.get("REACHY_VAD_OFF_MULT", 3.0))
VAD_ON_FLOOR = float(os.environ.get("REACHY_VAD_ON_FLOOR", 400))
VAD_OFF_FLOOR = float(os.environ.get("REACHY_VAD_OFF_FLOOR", 200))
VAD_HANGOVER_S = 0.8      # silence needed to close an utterance
VAD_MIN_UTTERANCE_S = 0.4  # ignore clicks and door slams
VAD_MAX_UTTERANCE_S = 15.0

# Wake phrase. "hey" is the default because it is short and easy to say, but
# note the trade-off: it is a common English interjection, so it will fire on
# ordinary conversation ("hey, did you see...") in a shared room. "hey reachy"
# is far more selective if false wakes become a nuisance.
#
# Variants exist because ASR does not render these cleanly: Parakeet was
# measured rendering "Reachy" as "Rich" / "Richie", and "hey" as "hay"/"eh".
WAKE_PHRASE = os.environ.get("REACHY_WAKE_PHRASE", "hey")
WAKE_PATTERNS = {
    # No bare "ay"/"eh": too loose, and they fire on filler speech.
    "hey": [r"\bhey\b", r"\bhay\b", r"\bhei\b"],
    "hey reachy": [
        r"\bhey\s+(reach|rich|ridge|retch)\w*\b",
        r"\breach(y|ie|ee)?\b", r"\brich(y|ie|ard)?\b", r"\breaching\b",
    ],
}
WAKE_RE = re.compile(
    "|".join(WAKE_PATTERNS.get(WAKE_PHRASE, [r"\b" + re.escape(WAKE_PHRASE) + r"\b"])),
    re.I,
)

# Said as "hey reachy ...", the name would otherwise survive into the question.
NAME_RE = re.compile(r"^(reach\w*|rich\w*|ridge\w*|retch\w*)\b[\s,.!?]*", re.I)

# REACHY_NO_WAKE=1 answers every utterance without a wake phrase. Useful for a
# first end-to-end test, and for a quiet room where you are the only speaker.
NO_WAKE = os.environ.get("REACHY_NO_WAKE", "") not in ("", "0", "false")

# Local wake detection keeps audio on this machine until the phrase is heard.
# Strongly preferred in any shared room: without it, every utterance in earshot
# is shipped to cloud ASR just to test for the wake word.
LOCAL_WAKE = os.environ.get("REACHY_LOCAL_WAKE", "1") not in ("", "0", "false")
LOCAL_WAKE_MODEL = os.environ.get("REACHY_LOCAL_WAKE_MODEL", "tiny.en")

# How long the agent keeps answering follow-ups without needing the wake word.
CONVERSATION_WINDOW_S = 25.0
VISION_INTERVAL_S = 3.0

# A tool call may itself trigger a follow-up call ("turn right and look
# happy"), so allow a few hops -- but cap them, or a model that keeps calling
# tools will spin.
MAX_TOOL_HOPS = int(os.environ.get("REACHY_MAX_TOOL_HOPS", 4))

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive desk robot with a camera, "
    "microphones, a movable head and antennas. You are speaking out loud, so "
    "reply in ONE or TWO short spoken sentences. Never use markdown, lists, or "
    "emoji. Be warm and concrete.\n\n"
    "You have tools that move your body. When the user asks you to move, turn, "
    "look somewhere, dance, or show an emotion, CALL THE TOOL -- do not merely "
    "describe what you would do. From the user's point of view, 'right' means "
    "your right, which is negative yaw. After a tool runs, confirm briefly in "
    "one short sentence.\n\n"
    "Only call a tool to MOVE. Questions about what you can see, or ordinary "
    "conversation, are answered in words with no tool call at all. Never "
    "invent a tool that is not in your list.\n\n"
    "You are given a live description of what your camera currently sees; use "
    "it naturally when relevant, and never mention that you were given a "
    "description."
)


# --------------------------------------------------------- NVIDIA speech (gRPC)


class Speech:
    """Riva ASR + TTS over the NVCF gRPC endpoint."""

    def __init__(self, api_key: str = NVIDIA_API_KEY):
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.api_key = api_key
        self._asr = None
        self._tts = None

    def _auth(self, function_id: str):
        import riva.client
        return riva.client.Auth(
            uri=RIVA_URI, use_ssl=True,
            metadata_args=[["function-id", function_id],
                           ["authorization", f"Bearer {self.api_key}"]],
        )

    @property
    def asr(self):
        if self._asr is None:
            import riva.client
            self._asr = riva.client.ASRService(self._auth(ASR_FUNCTION))
        return self._asr

    @property
    def tts(self):
        if self._tts is None:
            import riva.client
            self._tts = riva.client.SpeechSynthesisService(self._auth(TTS_FUNCTION))
        return self._tts

    def transcribe(self, pcm16: bytes) -> str:
        """16 kHz mono PCM -> text. Returns '' on silence."""
        import riva.client
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm16)
        cfg = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=RATE, language_code="en-US",
            max_alternatives=1, enable_automatic_punctuation=True,
        )
        resp = self.asr.offline_recognize(buf.getvalue(), cfg)
        return " ".join(
            r.alternatives[0].transcript for r in resp.results if r.alternatives
        ).strip()

    def synthesize(self, text: str) -> bytes:
        """text -> 44.1 kHz stereo WAV bytes, ready for the robot speaker."""
        import riva.client
        resp = self.tts.synthesize(
            text, voice_name=TTS_VOICE, language_code="en-US",
            sample_rate_hz=44100, encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
        mono = np.frombuffer(resp.audio, dtype="<i2").astype(np.float32)
        # Magpie returns ~30% of full scale, throwing away ~10 dB. The robot's
        # speaker is small, so normalise to just under clipping before upload;
        # this matters far more than the daemon's volume setting.
        peak = float(np.abs(mono).max())
        if peak > 0:
            gain = min(TTS_PEAK_TARGET * 32767.0 / peak, TTS_MAX_GAIN)
            mono = np.clip(mono * gain, -32768, 32767)
        mono = mono.astype("<i2")
        stereo = np.repeat(mono[:, None], 2, axis=1).ravel()   # robot expects 2ch
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(stereo.tobytes())
        return buf.getvalue()


# ----------------------------------------------------- NVIDIA vision + LLM (REST)


class Brain:
    """VLM captioning and LLM chat against the OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = NVIDIA_API_KEY):
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.http = httpx.Client(
            base_url=NVIDIA_REST, timeout=45.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def caption(self, jpeg: bytes) -> str:
        import base64
        b64 = base64.b64encode(jpeg).decode()
        r = self.http.post("/chat/completions", json={
            "model": VLM_MODEL, "max_tokens": 70, "temperature": 0.2,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text":
                    "Describe this scene in one short sentence: who is present, "
                    "what they are doing, and any notable objects."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        })
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    def _post(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body = {"model": LLM_MODEL, "max_tokens": 200, "temperature": 0.4,
                "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    @staticmethod
    def _confirm(done: list[str]) -> str:
        """Spoken confirmation built from what actually ran.

        Used when the model cannot produce a closing sentence. The physical
        action has already happened by then, so failing the whole turn would
        leave the robot moved but mute.
        """
        if not done:
            return "Done."
        if len(done) == 1:
            return f"Okay, {done[0]}."
        return "Okay, " + ", then ".join(done) + "."

    def reply(self, history: list[dict], world: str,
              on_tool=None) -> str:
        """One turn, resolving tool calls before answering.

        Loops because a request like "turn right and look happy" produces two
        calls; capped so a model that keeps calling tools cannot spin forever.
        """
        sys_msg = SYSTEM_PROMPT
        if world:
            sys_msg += f"\n\nYour camera currently sees: {world}"
        msgs = [{"role": "system", "content": sys_msg}] + list(history)
        tools = tool_schema()
        done: list[str] = []

        for _ in range(MAX_TOOL_HOPS):
            try:
                msg = self._post(msgs, tools)
            except httpx.HTTPStatusError as e:
                # The endpoint intermittently 500s on longer tool histories.
                # Retry once; if it still fails but tools already ran, speak
                # the confirmation rather than dropping the turn.
                if e.response.status_code < 500:
                    raise
                try:
                    msg = self._post(msgs, tools)
                except Exception:
                    if done:
                        return self._confirm(done)
                    raise
            calls = msg.get("tool_calls") or []
            if not calls:
                # Reasoning models put prose in reasoning_content, leaving
                # content empty; fall back so the robot still says something.
                text = ((msg.get("content") or "").strip()
                        or (msg.get("reasoning_content") or "").strip())
                # After a tool runs, models sometimes echo the raw result
                # instead of speaking. Spoken aloud that is literal JSON.
                if done and (not text or text.startswith(("{", "["))):
                    return self._confirm(done)
                # Models often quote a tool result verbatim; the quote marks
                # are noise once this is spoken aloud.
                return text.strip().strip('"').strip("'").strip()
            msgs.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                skill = SKILLS.get(name)
                if skill is None:
                    # List the real ones: models invent plausible tools
                    # (look_at, set_volume) and retry the same invention
                    # unless told what actually exists.
                    result = (f"no such tool {name!r}. Available tools: "
                              + ", ".join(t["function"]["name"] for t in tools)
                              + ". If none fit, just answer in words.")
                else:
                    try:
                        # Drop unexpected keys: models occasionally invent or
                        # nest arguments (gpt-oss wraps them in "degrees").
                        import inspect
                        ok = set(inspect.signature(skill).parameters)
                        result = skill(**{k: v for k, v in args.items() if k in ok})
                    except Exception as e:
                        result = f"{name} failed: {type(e).__name__}: {e}"
                if on_tool:
                    on_tool(name, args, result)
                if skill is not None and not str(result).startswith("no such"):
                    done.append(str(result))
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "name": name, "content": str(result)})
        return self._confirm(done)


# ---------------------------------------------------------- local wake detection


class LocalWake:
    """On-device wake-phrase detection with faster-whisper.

    The point is privacy, not accuracy: until the phrase is heard, no audio
    leaves this machine. Everything the mics pick up in a shared room -- other
    people, meetings, phone calls -- is transcribed locally, matched, and
    discarded. Only a matching utterance is forwarded to cloud ASR, which then
    does the accurate transcription.

    `tiny.en` is deliberate: it is ~75 MB, runs in well under a second on
    Apple Silicon CPU, and a wake phrase needs recall, not fidelity. The cloud
    model handles the words that actually matter.
    """

    def __init__(self, model_name: str = "", ):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            model_name or LOCAL_WAKE_MODEL, device="cpu", compute_type="int8"
        )

    def detect(self, pcm16: bytes) -> tuple[bool, str]:
        """Returns (wake_heard, local_transcript). Never leaves the machine."""
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text for s in segments).strip()
        return bool(WAKE_RE.search(text)), text


# ------------------------------------------------------------- robot A/V source


class RobotAV:
    """One gstreamer process: PCM on stdout, JPEG frames to a temp dir.

    The robot advertises a single WebRTC producer, so this must be the only
    consumer. Note both pads have to be drained — leaving one unlinked kills
    the pipeline in about half a second with `not-linked (-1)`.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.frames = Path(tempfile.mkdtemp(prefix="reachy_av_"))
        self.audio_q: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def producer_id() -> str | None:
        """Ask the robot's signalling server which camera producer to attach to.

        Spoken to directly rather than via the control server, so the agent
        runs standalone on any machine that can reach the robot.
        """
        async def ask() -> str | None:
            import websockets
            async with websockets.connect(SIGNALLING, open_timeout=6) as ws:
                await asyncio.wait_for(ws.recv(), 5)          # welcome frame
                await ws.send(json.dumps({"type": "list"}))
                resp = json.loads(await asyncio.wait_for(ws.recv(), 5))
                producers = resp.get("producers", [])
                return producers[0]["id"] if producers else None

        try:
            return asyncio.run(ask())
        except RuntimeError:
            # Already inside a running loop (mounted in FastAPI): use a thread.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(ask())).result(timeout=20)
        except Exception:
            return None

    def start(self) -> None:
        pid = self.producer_id()
        if not pid:
            raise RuntimeError(
                "no WebRTC producer found — is the daemon running and "
                "the control server (port 7788) up?"
            )
        cmd = [
            "gst-launch-1.0", "-q", "webrtcsrc", "name=ws",
            f"signaller::uri={SIGNALLING}",
            f"signaller::producer-peer-id={pid}",
            "ws.audio_0", "!", "queue", "!", "audioconvert", "!", "audioresample",
            "!", f"audio/x-raw,format=S16LE,rate={RATE},channels=1",
            "!", "fdsink", "fd=1",
            "ws.video_0", "!", "queue", "!", "videoconvert", "!", "videorate",
            "!", "video/x-raw,framerate=1/2", "!", "jpegenc", "quality=70",
            "!", "multifilesink", f"location={self.frames}/f_%05d.jpg", "max-files=4",
        ]
        # Keep gstreamer's stderr: a pipeline that dies (the classic being an
        # unlinked pad -> `not-linked (-1)`) is otherwise silently invisible.
        self.log_path = self.frames / "gst.log"
        self._errlog = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=self._errlog, bufsize=0
        )
        self._stop.clear()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        buf = b""
        while not self._stop.is_set():
            # A raw pipe returns short reads. Accumulate to an exact frame
            # instead of zero-padding, which would splice silence into the
            # stream and wreck both the VAD and the ASR input.
            data = self.proc.stdout.read(FRAME_BYTES - len(buf))
            if not data:
                break
            buf += data
            if len(buf) < FRAME_BYTES:
                continue
            chunk, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
            try:
                self.audio_q.put_nowait(chunk)
            except queue.Full:
                # Drop the oldest frame rather than block the reader; falling
                # behind on audio is better than stalling the pipeline.
                try:
                    self.audio_q.get_nowait()
                    self.audio_q.put_nowait(chunk)
                except queue.Empty:
                    pass

    def latest_frame(self) -> bytes | None:
        files = sorted(self.frames.glob("f_*.jpg"))
        # Skip the newest: multifilesink may still be writing it.
        for p in reversed(files[:-1] or files):
            try:
                data = p.read_bytes()
                if data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"):
                    return data
            except OSError:
                continue
        return None

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


# ------------------------------------------------------------- robot output


def play_wav(wav_bytes: bytes, name: str = "_reachy_voice.wav") -> float:
    """Upload a WAV to the robot and play it. Returns its duration in seconds."""
    with httpx.Client(base_url=ROBOT, timeout=60.0) as c:
        c.post("/api/media/sounds/upload",
               files={"file": (name, wav_bytes, "audio/wav")}).raise_for_status()
        c.post("/api/media/play_sound", json={"file": name}).raise_for_status()
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()


# ------------------------------------------------------------ robot skills
# Tools the LLM can call. Kept deliberately small and physical: each maps to
# one robot action with bounded arguments, so a mis-parsed argument cannot
# drive the hardware somewhere unreasonable.

EMOTIONS: list[str] = []       # filled lazily from the robot


def emotion_names() -> list[str]:
    global EMOTIONS
    if not EMOTIONS:
        try:
            r = httpx.get(
                f"{ROBOT}/api/move/recorded-move-datasets/list/"
                f"{quote(EMOTIONS_DATASET, safe='')}", timeout=15)
            data = r.json()
            EMOTIONS = data if isinstance(data, list) else data.get("moves", [])
        except Exception:
            EMOTIONS = []
    return EMOTIONS


def _clamp(v, lo: float, hi: float) -> float:
    """Clamp, coercing first.

    Models frequently emit numbers as JSON strings ("5" rather than 5), which
    would otherwise raise TypeError inside the comparison.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = 0.0
    return max(lo, min(hi, f))


# ASR gives plain words; the library uses names like "cheerful1". Without this
# map, "show me a happy emotion" finds nothing -- there is no entry called
# "happy" and fuzzy spelling matching cannot bridge a synonym.
EMOTION_ALIASES = {
    "happy": "cheerful1", "joy": "cheerful1", "joyful": "cheerful1",
    "glad": "cheerful1", "cheer": "cheerful1", "smile": "cheerful1",
    "excited": "enthusiastic1", "enthusiastic": "enthusiastic1",
    "sad": "sad1", "unhappy": "sad1", "upset": "downcast1", "cry": "sad2",
    "angry": "furious1", "mad": "rage1", "annoyed": "irritated1",
    "surprised": "surprised1", "shocked": "amazed1", "amazed": "amazed1",
    "curious": "curious1", "confused": "confused1", "puzzled": "uncertain1",
    "tired": "tired1", "sleepy": "sleep1", "sleep": "sleep1",
    "bored": "boredom1", "proud": "proud1", "laugh": "laughing1",
    "laughing": "laughing1", "funny": "laughing1",
    "dance": "dance1", "dancing": "dance1",
    "yes": "yes1", "nod": "yes1", "agree": "yes1",
    "no": "no1", "shake": "no1", "disagree": "no1",
    "hello": "welcoming1", "hi": "welcoming1", "greet": "welcoming1",
    "welcome": "welcoming1", "wave": "welcoming1",
    "scared": "scared1", "afraid": "fear1", "anxious": "anxiety1",
    "love": "loving1", "grateful": "grateful1", "thanks": "grateful1",
    "sorry": "oops1", "oops": "oops1", "think": "thoughtful1",
    "thinking": "thoughtful1", "calm": "calming1", "relief": "relief1",
    "success": "success1", "win": "success1", "shy": "shy1",
    "lonely": "lonely1", "disgusted": "disgusted1", "hello_there": "welcoming1",
}


def skill_move_head(yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                    roll_deg: float = 0.0, duration: float = 1.0) -> str:
    """Absolute head pose. Limits mirror the robot's usable range."""
    import math
    # Coerce up front, not just inside _clamp: the values are echoed back in
    # the confirmation string, and a JSON-string argument would blow up there.
    yaw_deg = _clamp(yaw_deg, -70, 70)
    pitch_deg = _clamp(pitch_deg, -35, 35)
    roll_deg = _clamp(roll_deg, -35, 35)
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    httpx.post(f"{ROBOT}/api/move/goto", timeout=15, json={
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                      "roll": roll, "pitch": pitch, "yaw": yaw},
        "duration": _clamp(duration, 0.3, 4.0), "interpolation": "minjerk",
    })
    # Phrased for speech, not for logs: the model frequently echoes a tool
    # result verbatim as its spoken reply, and "yaw -40, pitch 0" read aloud
    # sounds like a machine reciting telemetry.
    where = []
    if yaw_deg > 5:
        where.append("left")
    elif yaw_deg < -5:
        where.append("right")
    if pitch_deg < -5:
        where.append("up")
    elif pitch_deg > 5:
        where.append("down")
    return "looking " + " and ".join(where) if where else "looking straight ahead"


def skill_turn_body(yaw_deg: float = 0.0, duration: float = 1.5) -> str:
    import math
    yaw_deg = _clamp(yaw_deg, -160, 160)
    yaw = math.radians(yaw_deg)
    httpx.post(f"{ROBOT}/api/move/goto", timeout=15, json={
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                      "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "body_yaw": yaw, "duration": _clamp(duration, 0.5, 5.0),
        "interpolation": "minjerk",
    })
    side = "left" if yaw_deg > 0 else "right"
    return f"turned {abs(yaw_deg):.0f} degrees to the {side}"


def skill_play_emotion(name: str = "") -> str:
    """Emotion by name, with fuzzy matching -- ASR will not produce 'curious1'."""
    import difflib
    names = emotion_names()
    if not names:
        return "emotion library unavailable"
    want = (name or "").strip().lower().replace(" ", "_")
    match = None
    if want in names:
        match = want
    elif want in EMOTION_ALIASES and EMOTION_ALIASES[want] in names:
        match = EMOTION_ALIASES[want]                       # synonym
    elif want:
        # 'curious' -> 'curious1'; then fall back to nearest spelling.
        prefixed = [n for n in names if n.lower().startswith(want)]
        close = difflib.get_close_matches(want, names, n=1, cutoff=0.6)
        match = prefixed[0] if prefixed else (close[0] if close else None)
    if not match:
        return f"no emotion matching {name!r}"
    play_emotion(match)
    # Strip the trailing index: "cheerful1" spoken aloud is odd.
    return f"showing {re.sub(r'[0-9]+$', '', match).replace('_', ' ')}"


def skill_reset_pose() -> str:
    httpx.post(f"{ROBOT}/api/move/goto", timeout=15, json={
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                      "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "antennas": [0.0, 0.0], "body_yaw": 0.0, "duration": 1.5,
    })
    return "back to neutral"


def skill_run_agent(task: str = "") -> str:
    """Hand a task to an external agent CLI (files, web, shell, etc.).

    Disabled unless REACHY_AGENT_ENABLED=1. This turns anything sayable within
    earshot of the robot into a command on your machine, with no authentication
    -- so it is opt-in, not a default.
    """
    if not AGENT_ENABLED:
        return ("the agent tool is disabled; set REACHY_AGENT_ENABLED=1 "
                "to allow it")
    if not task.strip():
        return "no task given"
    try:
        proc = subprocess.run(
            AGENT_CMD + [task], capture_output=True, text=True,
            timeout=AGENT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"the agent did not finish within {AGENT_TIMEOUT} seconds"
    except FileNotFoundError:
        return f"agent command not found: {AGENT_CMD[0]}"
    out = (proc.stdout or proc.stderr or "").strip()
    return out[:AGENT_MAX_CHARS] or "the agent returned nothing"


SKILLS = {
    "move_head": skill_move_head,
    "turn_body": skill_turn_body,
    "play_emotion": skill_play_emotion,
    "reset_pose": skill_reset_pose,
    "run_agent": skill_run_agent,
}


def tool_schema() -> list[dict]:
    tools = [
        {"type": "function", "function": {
            "name": "move_head",
            "description": "Turn or tilt the robot's head to an absolute pose. "
                           "Positive yaw is LEFT, negative yaw is RIGHT. "
                           "Positive pitch looks DOWN, negative pitch looks UP "
                           "(verified on hardware). You must supply a non-zero "
                           "value for the axis you are asked to move: to look "
                           "up use pitch_deg -25, to look down use 25, to look "
                           "left use yaw_deg 40, to look right use -40. Zero "
                           "means centred, so never send 0 for the axis the "
                           "user asked you to move.",
            "parameters": {"type": "object", "properties": {
                "yaw_deg": {"type": "number", "description": "-70 (right) to 70 (left)"},
                "pitch_deg": {"type": "number", "description": "-35 (up) to 35 (down)"},
                "roll_deg": {"type": "number", "description": "-35 to 35, head tilt"},
            }, "required": ["yaw_deg"]}}},
        {"type": "function", "function": {
            "name": "turn_body",
            "description": "Rotate the whole body. Positive is LEFT, negative "
                           "is RIGHT. Use for large turns beyond head range.",
            "parameters": {"type": "object", "properties": {
                "yaw_deg": {"type": "number", "description": "-160 to 160"},
            }, "required": ["yaw_deg"]}}},
        {"type": "function", "function": {
            "name": "play_emotion",
            "description": "Play an expressive animation. Use when asked to "
                           "show an emotion, dance, nod yes, or shake no.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string",
                         "description": "e.g. happy, curious, sad, dance, yes, "
                                        "no, surprised, proud, laughing"},
            }, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "reset_pose",
            "description": "Return the head, body and antennas to neutral.",
            "parameters": {"type": "object", "properties": {}}}},
    ]
    if AGENT_ENABLED:
        tools.append({"type": "function", "function": {
            "name": "run_agent",
            "description": "Delegate a computer task to an external coding "
                           "agent: search the web, read or check files, run "
                           "commands, look something up. Pass the request in "
                           "plain English. Slow (seconds to minutes).",
            "parameters": {"type": "object", "properties": {
                "task": {"type": "string",
                         "description": "the task, in plain English"},
            }, "required": ["task"]}}})
    return tools


class Motion:
    """Body language for the conversation states.

    Runs in its own thread and posts short `goto` moves, so the head and
    antennas keep moving while the main loop is blocked on ASR or the LLM.
    Ranges are kept well inside the limits used by server.py's dance steps.
    """

    MODES = ("idle", "listening", "thinking", "speaking")

    def __init__(self):
        self.mode = "idle"
        self.running = False
        self._t: threading.Thread | None = None
        self._http = httpx.Client(base_url=ROBOT, timeout=5.0)
        self._i = 0

    def set(self, mode: str) -> None:
        if mode in self.MODES and mode != self.mode:
            self.mode = mode
            self._i = 0

    def _goto(self, *, z=0.0, roll=0.0, pitch=0.0, yaw=0.0,
              antennas=(0.0, 0.0), duration=0.6) -> None:
        try:
            self._http.post("/api/move/goto", json={
                "head_pose": {"x": 0.0, "y": 0.0, "z": z,
                              "roll": roll, "pitch": pitch, "yaw": yaw},
                "antennas": list(antennas),
                "duration": duration, "interpolation": "minjerk",
            })
        except Exception:
            pass  # never let body language break the conversation

    def _loop(self) -> None:
        import math
        while self.running:
            i, m = self._i, self.mode
            self._i += 1
            if m == "speaking":
                # Antennas flick and the head bobs while talking -- the cue
                # that reads most clearly as "this thing is addressing me".
                s = math.sin(i * 1.1)
                self._goto(z=0.006 * s, pitch=-0.05 + 0.07 * s,
                           yaw=0.16 * math.sin(i * 0.7), roll=0.05 * s,
                           antennas=(0.7 + 0.5 * s, 0.7 - 0.5 * s), duration=0.38)
                time.sleep(0.34)
            elif m == "listening":
                # Alert and still: head up a touch, antennas raised, minimal
                # drift so it reads as attention rather than fidgeting.
                self._goto(z=0.004, pitch=-0.13, yaw=0.10 * math.sin(i * 0.35),
                           antennas=(0.85, 0.85), duration=0.9)
                time.sleep(0.85)
            elif m == "thinking":
                self._goto(z=-0.004, pitch=0.16, roll=0.13,
                           antennas=(-0.35, 0.45), duration=0.8)
                time.sleep(0.75)
            else:  # idle -- slow breathing
                self._goto(z=0.008 * math.sin(i * 0.5), pitch=0.02,
                           yaw=0.12 * math.sin(i * 0.23),
                           antennas=(0.1 * math.sin(i * 0.4),) * 2, duration=2.0)
                time.sleep(1.9)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self.running = False
        self._goto(duration=1.2)  # return to neutral


def play_emotion(name: str) -> None:
    try:
        httpx.post(
            f"{ROBOT}/api/move/play/recorded-move-dataset/"
            f"{quote(EMOTIONS_DATASET, safe='')}/{name}",
            timeout=10.0,
        )
    except Exception:
        pass  # expressive garnish; never let it break the conversation


# ------------------------------------------------------------------ the agent


class VoiceAgent:
    def __init__(self):
        self.speech = Speech()
        self.brain = Brain()
        self.av = RobotAV()
        self.history: list[dict] = []
        self.world = ""
        self.world_at = 0.0
        self.events: list[dict] = []
        self.running = False
        self.awake_until = 0.0
        self._mute_until = 0.0
        self._thread: threading.Thread | None = None
        # ~30 s of frame energies, used to estimate the room noise floor
        self.noise: deque[float] = deque(maxlen=300)
        self.motion = Motion()
        self.level = 0.0        # latest frame rms, for the GUI meter
        self.gate = 0.0         # current VAD open threshold
        self.state = "idle"     # idle | listening | thinking | speaking
        self.local_wake: LocalWake | None = None
        if LOCAL_WAKE and not NO_WAKE:
            try:
                self.local_wake = LocalWake()
            except Exception as e:
                # Loud, not silent: without this the room goes to the cloud.
                print(f"WARNING: local wake unavailable ({type(e).__name__}: {e}).\n"
                      f"         Every utterance will be sent to cloud ASR to test\n"
                      f"         for the wake word. Install with:\n"
                      f"           .venv/bin/pip install faster-whisper\n", flush=True)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.motion.set(state)
            self.emit("state", state=state)

    # -- event log the UI can poll -------------------------------------
    def emit(self, kind: str, **data) -> None:
        ev = {"t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        del self.events[:-200]
        print(f"[{kind}] " + " ".join(f"{k}={v!r}" for k, v in data.items()), flush=True)

    # -- background vision --------------------------------------------
    def _vision_loop(self) -> None:
        while self.running:
            time.sleep(VISION_INTERVAL_S)
            frame = self.av.latest_frame()
            if not frame:
                continue
            try:
                cap = self.brain.caption(frame)
                if cap:
                    self.world, self.world_at = cap, time.time()
                    self.emit("vision", caption=cap)
            except Exception as e:
                self.emit("error", where="vision", detail=f"{type(e).__name__}: {e}")

    # -- utterance segmentation ---------------------------------------
    def _next_utterance(self) -> bytes | None:
        """Block until a speech segment completes; return its PCM."""
        voiced: list[bytes] = []
        silence = 0.0
        recent: list[float] = []
        frames_seen = 0
        next_report = time.time() + 10.0
        while self.running:
            try:
                frame = self.av.audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if time.time() < self._mute_until:
                voiced.clear()
                silence = 0.0
                continue
            rms = float(np.frombuffer(frame, dtype="<i2").astype(np.float32).std())

            # Estimate the room floor from a low percentile of ALL recent
            # frames, not just non-voiced ones. Sampling only when idle looks
            # tidier but deadlocks: if the gate latches open the estimator
            # stops updating, the floor drifts to a speech-level value, and
            # nothing can ever close the utterance again. A p20 over ~30 s is
            # robust because natural speech is full of gaps.
            # Measured in this room: silence ~30, speech bursts 400-13000.
            self.noise.append(rms)
            floor = (float(np.percentile(self.noise, 20))
                     if len(self.noise) >= 30 else 40.0)
            on_thr = max(floor * VAD_ON_MULT, VAD_ON_FLOOR)
            off_thr = max(floor * VAD_OFF_MULT, VAD_OFF_FLOOR)

            self.level, self.gate = rms, on_thr
            if voiced and self.state == "idle":
                self._set_state("listening")
            recent.append(rms)
            frames_seen += 1
            if time.time() >= next_report:
                arr = np.array(recent or [0.0])
                self.emit("audio", frames=frames_seen, noise_floor=round(floor),
                          median=round(float(np.median(arr))),
                          peak=round(float(arr.max())),
                          on=round(on_thr), off=round(off_thr),
                          in_utterance=len(voiced))
                recent.clear()
                frames_seen = 0
                next_report = time.time() + 10.0
            if not voiced:
                if rms >= on_thr:
                    voiced.append(frame)
            else:
                voiced.append(frame)
                silence = silence + FRAME_MS / 1000 if rms < off_thr else 0.0
                dur = len(voiced) * FRAME_MS / 1000
                if silence >= VAD_HANGOVER_S or dur >= VAD_MAX_UTTERANCE_S:
                    why = "silence" if silence >= VAD_HANGOVER_S else "maxlen"
                    if dur < VAD_MIN_UTTERANCE_S:
                        voiced.clear()
                        silence = 0.0
                        continue
                    self.emit("segment", secs=round(dur, 1), why=why)
                    return b"".join(voiced)
        return None

    # -- one conversational turn ---------------------------------------
    def _respond(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        del self.history[:-12]
        self._set_state("thinking")
        t0 = time.time()

        def on_tool(name, args, result):
            self.emit("tool", name=name, args=args, result=str(result)[:120])

        try:
            answer = self.brain.reply(self.history, self.world, on_tool=on_tool)
        except Exception as e:
            self.emit("error", where="llm", detail=f"{type(e).__name__}: {e}")
            self._set_state("idle")
            return
        if not answer:
            answer = "Sorry, I didn't catch that."
        self.history.append({"role": "assistant", "content": answer})
        self.emit("reply", text=answer, llm_s=round(time.time() - t0, 2))
        try:
            wav = self.speech.synthesize(answer)
            self._set_state("speaking")
            dur = play_wav(wav)
            # The XMOS board does hardware AEC so the robot will not hear
            # itself, but muting keeps our own VAD from chasing the tail.
            self._mute_until = time.time() + dur + 0.3
            self.awake_until = self._mute_until + CONVERSATION_WINDOW_S
            # Keep the talking animation running for the length of the audio,
            # then settle. play_sound returns as soon as playback starts.
            threading.Timer(dur, lambda: self._set_state("idle")).start()
        except Exception as e:
            self.emit("error", where="tts", detail=f"{type(e).__name__}: {e}")
            self._set_state("idle")

    # -- main loop ------------------------------------------------------
    def _run(self) -> None:
        try:
            self.av.start()
        except Exception as e:
            self.emit("error", where="av", detail=str(e))
            self.running = False
            return
        self.motion.start()
        self.emit("ready",
                  wake="everything" if NO_WAKE else f"say '{WAKE_PHRASE}'")
        threading.Thread(target=self._vision_loop, daemon=True).start()
        while self.running:
            pcm = self._next_utterance()
            if not pcm:
                continue
            secs = len(pcm) / 2 / RATE
            in_window = time.time() < self.awake_until

            # ---- local gate -------------------------------------------------
            # Nothing is sent anywhere until this passes. In a shared room this
            # is the difference between "the robot ignores your meeting" and
            # "your meeting is streamed to a cloud ASR to check for a keyword".
            if not (NO_WAKE or in_window) and self.local_wake is not None:
                t0 = time.time()
                try:
                    hit, local_text = self.local_wake.detect(pcm)
                except Exception as e:
                    self.emit("error", where="local_wake",
                              detail=f"{type(e).__name__}: {e}")
                    continue
                if not hit:
                    self.emit("ignored", secs=round(secs, 1),
                              local=local_text[:70], took=round(time.time() - t0, 2))
                    continue
                self.emit("wake", local=local_text[:70],
                          took=round(time.time() - t0, 2))

            try:
                text = self.speech.transcribe(pcm)
            except Exception as e:
                self.emit("error", where="asr", detail=f"{type(e).__name__}: {e}")
                continue
            if not text:
                # Worth logging: a segment that reaches ASR and comes back
                # empty means the VAD is firing on noise, not speech.
                self.emit("empty", secs=round(secs, 1))
                self._set_state("idle")
                continue

            m = None if NO_WAKE else WAKE_RE.search(text)
            # With the local gate the phrase is already confirmed; cloud ASR may
            # render it differently, so do not require a second match.
            gated_locally = self.local_wake is not None and not (NO_WAKE or in_window)
            if not (m or in_window or NO_WAKE or gated_locally):
                # Understood perfectly, just not addressed to us. This must NOT
                # be reported as `heard`: showing it as an accepted turn that
                # then gets no reply reads as a hang rather than a design.
                self.emit("unaddressed", secs=round(secs, 1), text=text[:120])
                self._set_state("idle")
                continue
            self.emit("heard", text=text, secs=round(secs, 1))
            if (m or gated_locally) and not in_window:
                # Strip the wake phrase; whatever follows is the actual query.
                # If only the local gate matched, cloud ASR may have spelled the
                # phrase differently, so fall back to the whole utterance.
                query = (text[m.end():] if m else text).lstrip(" ,.!?").strip()
                query = NAME_RE.sub("", query).strip()
                if not query:
                    self.awake_until = time.time() + CONVERSATION_WINDOW_S
                    play_emotion("attentive1")
                    try:
                        self._mute_until = time.time() + play_wav(
                            self.speech.synthesize("Yes?")) + 0.3
                    except Exception:
                        pass
                    continue
            else:
                query = text
            self._respond(query)
        self.av.stop()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self.motion.stop()
        self.av.stop()


# --------------------------------------------------------------- HTTP surface

router = APIRouter(prefix="/api/voice")
_agent: VoiceAgent | None = None


class SayReq(BaseModel):
    text: str


@router.post("/start")
async def voice_start():
    global _agent
    if _agent is None:
        _agent = VoiceAgent()
    _agent.start()
    return {"status": "started"}


@router.post("/stop")
async def voice_stop():
    if _agent:
        _agent.stop()
    return {"status": "stopped"}


@router.post("/wake")
async def voice_wake():
    """Open the conversation window from the GUI, no wake phrase needed.

    Equivalent to saying "hey reachy" -- the next thing you say is treated as
    a question. Useful when the wake word is awkward, or when cloud ASR keeps
    mangling it.
    """
    if not _agent or not _agent.running:
        return JSONResponse({"error": "agent not running"}, 409)
    _agent.awake_until = time.time() + CONVERSATION_WINDOW_S
    _agent.emit("wake", local="(from GUI)", took=0.0)
    play_emotion("attentive1")
    return {"status": "awake", "seconds": CONVERSATION_WINDOW_S}


@router.get("/status")
async def voice_status():
    if not _agent:
        return {"running": False, "state": "off"}
    return {
        "running": _agent.running,
        "state": _agent.state,
        "world": _agent.world,
        "world_age_s": round(time.time() - _agent.world_at, 1) if _agent.world_at else None,
        "awake": time.time() < _agent.awake_until,
        "wake_phrase": WAKE_PHRASE,
        "turns": len(_agent.history) // 2,
        "level": round(_agent.level), "gate": round(_agent.gate),
        "local_wake": _agent.local_wake is not None,
        "no_wake": NO_WAKE,
        "llm": LLM_MODEL, "vlm": VLM_MODEL, "tts": TTS_VOICE,
    }


@router.get("/stream")
async def voice_stream():
    """MJPEG of the frames the agent is already capturing.

    Deliberately not a second WebRTC connection: the robot advertises one
    producer, so the GUI has to reuse the agent's frames rather than compete
    with it for the camera.
    """
    async def gen():
        last = None
        while _agent and _agent.running:
            frame = _agent.av.latest_frame()
            if frame and frame != last:
                last = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() +
                       b"\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.4)
    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/events")
async def voice_events(since: float = 0.0):
    evs = [e for e in (_agent.events if _agent else []) if e["t"] > since]
    return {"events": evs, "now": time.time()}


@router.post("/say")
async def voice_say(req: SayReq):
    """Speak arbitrary text through Magpie — handy for testing without talking."""
    sp = Speech()
    dur = await asyncio.to_thread(lambda: play_wav(sp.synthesize(req.text)))
    return {"status": "ok", "seconds": round(dur, 2)}


# ------------------------------------------------------------------ standalone

def preflight() -> bool:
    """Print a PASS/FAIL checklist. A silent startup is impossible to debug."""
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<22} {detail}", flush=True)

    print("Reachy voice agent — preflight", flush=True)
    line("NVIDIA_API_KEY", bool(NVIDIA_API_KEY),
         f"{NVIDIA_API_KEY[:9]}..." if NVIDIA_API_KEY else "not set (export it)")
    try:
        import riva.client  # noqa: F401
        line("riva client", True)
    except Exception as e:
        line("riva client", False, f"{e} — pip install nvidia-riva-client")
    if NO_WAKE:
        line("local wake", True, "SKIPPED — REACHY_NO_WAKE=1 replies to everything")
    else:
        try:
            import faster_whisper  # noqa: F401
            line("local wake", True, f"faster-whisper {LOCAL_WAKE_MODEL} (audio stays local)")
        except Exception:
            line("local wake", False,
                 "not installed — every utterance would go to cloud ASR. "
                 "pip install faster-whisper")
    try:
        r = httpx.get(f"{ROBOT}/api/daemon/status", timeout=5)
        line("robot daemon", r.status_code == 200, f"{ROBOT}")
    except Exception as e:
        line("robot daemon", False,
             f"{ROBOT} unreachable: {e}\n"
             f"         If this is 'No route to host', grant your terminal app\n"
             f"         Local Network access: System Settings > Privacy &\n"
             f"         Security > Local Network")
    pid = RobotAV.producer_id()
    line("webrtc producer", bool(pid),
         pid or "none — is the control server on :7788 up?")
    try:
        subprocess.run(["gst-launch-1.0", "--version"], capture_output=True, timeout=10)
        line("gstreamer", True)
    except Exception as e:
        line("gstreamer", False, f"{e} — brew install gstreamer")
    # -x matches the executable name exactly. `pgrep -f` would also match any
    # shell whose command line merely mentions gst-launch-1.0, including the
    # one running this check.
    busy = subprocess.run(["pgrep", "-x", "gst-launch-1.0"], capture_output=True)
    n = len(busy.stdout.split())
    line("producer free", n == 0,
         "ok" if n == 0 else f"{n} gst process(es) already running — "
         "close the Robot Eyes panel or kill them")
    print(flush=True)
    return ok


def main() -> int:
    """Console entry point (`reachy-voice`)."""
    if not preflight():
        print("Preflight failed — fix the FAIL lines above and rerun.", flush=True)
        return 1
    agent = VoiceAgent()
    agent.start()
    print(f"Listening. Say '{WAKE_PHRASE} ...'   Ctrl-C to stop.", flush=True)
    try:
        while agent.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
