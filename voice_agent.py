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

# Measured latencies: mistral-nemotron 0.41s, llama-3.1-8b 1.15s, 70b timed out.
LLM_MODEL = os.environ.get("REACHY_LLM", "mistralai/mistral-nemotron")
VLM_MODEL = os.environ.get("REACHY_VLM", "nvidia/nemotron-nano-12b-v2-vl")

TTS_FUNCTION = FN_TTS_MAGPIE
TTS_VOICE = "Magpie-Multilingual"
ASR_FUNCTION = FN_ASR_PARAKEET

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# Peak-normalise TTS to just under clipping. Gain is capped so a near-silent
# synthesis result cannot be amplified into a wall of noise.
TTS_PEAK_TARGET = float(os.environ.get("REACHY_TTS_PEAK", 0.97))
TTS_MAX_GAIN = float(os.environ.get("REACHY_TTS_MAX_GAIN", 8.0))

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

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive desk robot with a camera and "
    "microphones. You are speaking out loud, so reply in ONE or TWO short "
    "spoken sentences. Never use markdown, lists, or emoji. Be warm and "
    "concrete. You are given a live description of what your camera currently "
    "sees; use it naturally when it is relevant, and do not mention that you "
    "were given a description."
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

    def reply(self, history: list[dict], world: str) -> str:
        sys_msg = SYSTEM_PROMPT
        if world:
            sys_msg += f"\n\nYour camera currently sees: {world}"
        r = self.http.post("/chat/completions", json={
            "model": LLM_MODEL, "max_tokens": 120, "temperature": 0.7,
            "messages": [{"role": "system", "content": sys_msg}] + history,
        })
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # Reasoning models put prose in reasoning_content and leave content empty.
        return (msg.get("content") or "").strip()


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
        try:
            answer = self.brain.reply(self.history, self.world)
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
