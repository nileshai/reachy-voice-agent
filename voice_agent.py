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

# Benchmarked across every chat model on build.nvidia.com reachable from this
# account (median / max seconds over repeated streamed replies):
#   nemotron-3-nano-30b-a3b   0.46 / 0.76, ttft 0.27, tools OK   <- default
#   llama-3.1-8b-instruct     0.81 / 1.18, ttft 0.31, tools OK
#   gpt-oss-20b               1.32 / 1.50, ttft 0.94, tools OK
#   nemotron-nano-9b-v2       2.62 / 2.81, no tool calls
#   nemotron-3.5-lightning    3.43 / 3.87, ttft 2.66 -- best prose, too slow
#   nemotron-mini-4b          0.39 / 0.48 but never calls tools
#   mistral-nemotron          returns tool_calls: [] -- explains, never acts
#   gpt-oss-120b              nests tool arguments wrongly
#   nemotron-3-super-120b     correct calls but 25-30s per reply
#
# The default is a 30B mixture-of-experts with ~3B active parameters, which is
# also the right shape for a DGX Spark: the whole model fits in 128 GB unified
# memory while only the active experts cost compute per token. Served locally
# there, the network round-trip disappears from every turn.
LLM_MODEL = os.environ.get("REACHY_LLM", "nvidia/nemotron-3-nano-30b-a3b")

# Nemotron reasoning models emit their chain-of-thought into `content` unless
# thinking is switched off -- which would be spoken aloud verbatim ("Here's a
# thinking process: 1. Analyze User Input..."). Sent only to models that
# understand it; an unknown key is not universally ignored.
def _no_think_kwargs(model: str) -> dict:
    m = model.lower()
    if "nemotron" in m and any(k in m for k in ("lightning", "3.5", "-3-", "super", "nano")):
        return {"chat_template_kwargs": {"thinking": False}}
    return {}


LLM_EXTRA = _no_think_kwargs(LLM_MODEL)
# Vision. Hosted VLMs go down without warning -- nemotron-nano-12b-v2-vl was
# timing out on every call for hours while other VLMs answered in ~1s on the
# identical payload -- so the captioner rotates to the next model after
# repeated failures instead of blocking the whole vision loop.
VLM_MODEL = os.environ.get("REACHY_VLM", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
VLM_FALLBACKS = [m for m in [
    VLM_MODEL,
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/nemotron-nano-12b-v2-vl",
] if m]
VLM_FAILS_BEFORE_ROTATE = int(os.environ.get("REACHY_VLM_ROTATE_AFTER", 3))

TTS_FUNCTION = FN_TTS_MAGPIE
TTS_VOICE = "Magpie-Multilingual"
ASR_FUNCTION = FN_ASR_PARAKEET

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# TTS loudness. TTS_TARGET_RMS is the average level aimed for as a fraction of
# full scale; 0.16 is about -16 dBFS, loud for speech without sounding crushed.
# Raise towards 0.22 for a noisy room, lower to 0.10 if it sounds harsh. Gain
# is capped so a near-silent synthesis cannot be amplified into a wall of noise.
# 22.05 kHz mono: a quarter the bytes of 44.1 kHz stereo, indistinguishable on
# this speaker, and measurably quicker to upload.
TTS_RATE = int(os.environ.get("REACHY_TTS_RATE", 22050))
TTS_PEAK_TARGET = float(os.environ.get("REACHY_TTS_PEAK", 0.97))
TTS_TARGET_RMS = float(os.environ.get("REACHY_TTS_RMS", 0.16))
TTS_MAX_GAIN = float(os.environ.get("REACHY_TTS_MAX_GAIN", 12.0))

# External agent delegation (files, web, shell). OFF by default and
# deliberately so: with it on, anything said within earshot of the robot
# becomes a command executed on this machine, unauthenticated. Enable only
# when you are alone and understand what the agent CLI is permitted to do.
AGENT_ENABLED = os.environ.get("REACHY_AGENT_ENABLED", "") not in ("", "0", "false")
# Headless agents grant no tool permissions by default, so web search is
# declined unless it is named explicitly.
AGENT_CMD = shlex.split(os.environ.get(
    "REACHY_AGENT_CMD", "claude -p --allowedTools WebSearch,WebFetch"))
AGENT_TIMEOUT = float(os.environ.get("REACHY_AGENT_TIMEOUT", 120))
AGENT_MAX_CHARS = int(os.environ.get("REACHY_AGENT_MAX_CHARS", 700))

# Web search. Unlike the agent tool this is read-only and cannot touch the
# machine, so it is enabled whenever a backend is configured. Tavily is
# preferred: it returns a synthesised answer as well as snippets, which suits
# a spoken reply. build.nvidia.com hosts no general web search -- its "search"
# NIMs are protein-MSA, OCR and 3D-asset retrieval.
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
BRAVE_KEY = os.environ.get("BRAVE_API_KEY", "")
WEB_ENABLED = os.environ.get("REACHY_WEB_ENABLED", "1") not in ("", "0", "false")
WEB_TIMEOUT = float(os.environ.get("REACHY_WEB_TIMEOUT", 12))
WEB_MAX_RESULTS = int(os.environ.get("REACHY_WEB_MAX_RESULTS", 4))
WEB_MAX_CHARS = int(os.environ.get("REACHY_WEB_MAX_CHARS", 900))

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
# Absolute backstops. Kept low enough that a quiet room does not silently
# raise the bar out of reach: measured with the room floor at ~20, a 400 gate
# demanded speech 20x the noise level, and normal talking at a normal distance
# peaked at 275 -- so the VAD never opened at all and the agent looked wedged.
VAD_ON_FLOOR = float(os.environ.get("REACHY_VAD_ON_FLOOR", 220))
VAD_OFF_FLOOR = float(os.environ.get("REACHY_VAD_OFF_FLOOR", 110))

# A segment must also be sustained, not just have one loud spike. A door slam
# or chair scrape clears the gate on peak alone; speech keeps energy up across
# most of its frames. This is checked before calling ASR, so noise costs
# nothing and stops filling the transcript with "no speech".
VAD_MIN_MEDIAN_MULT = float(os.environ.get("REACHY_VAD_MEDIAN_MULT", 0.75))
# Silence needed to close an utterance. This is dead time the user feels on
# every single turn, so it is kept as short as endpointing allows.
VAD_HANGOVER_S = float(os.environ.get("REACHY_VAD_HANGOVER", 0.40))
# A bare "hey" measures ~0.4s, exactly the old cutoff, so the wake word was
# being discarded before the detector ever saw it -- which reads as "the wake
# word works sometimes". The detector itself is robust: 0/9 missed at quarter
# volume and 8 dB SNR. Sustained-energy screening (below) is what rejects
# clicks now, so this can be short.
VAD_MIN_UTTERANCE_S = float(os.environ.get("REACHY_VAD_MIN_UTTERANCE", 0.22))
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

# Conversation lifecycle, for an agent expected to sit in a room for days.
HISTORY_TURNS = int(os.environ.get("REACHY_HISTORY_TURNS", 12))
HISTORY_MAX_CHARS = int(os.environ.get("REACHY_HISTORY_MAX_CHARS", 4000))
SESSION_RESET_S = float(os.environ.get("REACHY_SESSION_RESET", 600))
VISION_INTERVAL_S = float(os.environ.get("REACHY_VISION_INTERVAL", 3.0))
VISION_MAX_BACKOFF_S = float(os.environ.get("REACHY_VISION_MAX_BACKOFF", 60))
# Captioning is background context, never in the critical path of a reply, so
# it should give up quickly rather than tie up a worker for 45s.
VLM_TIMEOUT = float(os.environ.get("REACHY_VLM_TIMEOUT", 12))

# A tool call may itself trigger a follow-up call ("turn right and look
# happy"), so allow a few hops -- but cap them, or a model that keeps calling
# tools will spin.
MAX_TOOL_HOPS = int(os.environ.get("REACHY_MAX_TOOL_HOPS", 3))
# One lookup answers one question; batching several is pure latency.
MAX_CALLS_PER_TURN = int(os.environ.get("REACHY_MAX_CALLS", 1))

# Tools slow enough that the robot should say something before starting.
# Transcribe concurrently with speech once a conversation is open.
STREAM_ASR = os.environ.get("REACHY_STREAM_ASR", "1") not in ("", "0", "false")

# Barge-in: talking over the robot cuts it off mid-sentence. Safe here because
# the XMOS board cancels the speaker in hardware -- measured mic level during
# the robot's own full-volume speech was 35, below the room floor.
BARGE_IN = os.environ.get("REACHY_BARGE_IN", "1") not in ("", "0", "false")
# Barge-in needs a higher bar than normal speech onset: while speaking the
# robot is also moving, and the mics hear the servos. Measured with motion
# running, a majority of frames cleared the ordinary gate.
BARGE_MIN_FRAMES = int(os.environ.get("REACHY_BARGE_FRAMES", 3))   # x100 ms
BARGE_MULT = float(os.environ.get("REACHY_BARGE_MULT", 2.5))
BARGE_FLOOR = float(os.environ.get("REACHY_BARGE_FLOOR", 2500))

# Deaf window after playback starts, covering the click of play_sound. This
# used to span the whole clip, which is what made barge-in impossible.
PLAY_GUARD_S = float(os.environ.get("REACHY_PLAY_GUARD", 0.35))

# Spoken while a slow lookup runs. Deliberately vague about progress -- the
# agent gives no signal, so anything specific would be a lie -- and varied,
# because the same phrase repeated is worse than silence.
FILLER_LINES = [
    "Still digging, hang on.",
    "Getting there, this one is taking a moment.",
    "Almost have it.",
    "Nearly done now.",
    "Just pulling the last of it together.",
]
FILLER_GAP_S = float(os.environ.get("REACHY_FILLER_GAP", 1.2))
# Wait this long before saying anything. A Tavily lookup often returns inside
# it, and silence beats an announcement that outlasts the search.
FILLER_DELAY_S = float(os.environ.get("REACHY_FILLER_DELAY", 1.3))

SLOW_TOOLS = {"run_agent"}

# Tools that return information to be conveyed, rather than an action taken.
INFO_TOOLS = {"web_search", "run_agent"}

# How long a warmed gRPC channel is trusted before re-warming.
WARM_IDLE_S = float(os.environ.get("REACHY_WARM_IDLE", 90))

# The camera caption is only supplied when the question is plausibly about
# what the robot can see. Keeping it in the prompt unconditionally makes a
# small model occasionally answer "what's the date?" by describing the room --
# measured at roughly 2 in 8 on llama-3.1-8b. Gating on intent removes that
# failure mode outright rather than hoping a prompt instruction holds.
# Tools are offered ONLY when movement is actually intended. Left always on,
# llama-3.1-8b answers "can you tell me what to do?" with a play_emotion call
# and empty content -- which then surfaces as a terse "Confirmed." Gating on
# intent keeps conversation conversational, and is faster too: the tool schema
# is a few hundred prompt tokens that most turns do not need.
MOVE_RE = re.compile(
    r"\b(turn|rotate|spin|tilt|nod|shake|dance|wave|reset|neutral|antennas?)\b"
    r"|\bre-?cent(er|re)\b"
    r"|\blook\s+(up|down|left|right|straight|ahead|forward|at|away|around)\b"
    r"|\bmove\s+(your|the|head|body|left|right|up|down)\b"
    r"|\bface\s+(me|left|right|forward|the)\b"
    r"|\d+\s*degrees?\b"
    r"|\b(show|give|do|play|make)\b[^.?!]{0,25}\b(emotion|expression|face|move|"
    r"animation|dance)\b"
    # "act happy" is a request; "I feel happy" is not, hence the leading verb.
    r"|\b(act|seem|pretend to be|look)\s+(happy|sad|angry|curious|surprised|"
    r"proud|scared|shy|bored|excited|tired|sleepy|confused)\b", re.I)

# Questions that need current information the model cannot know. Offering the
# search tool only here keeps ordinary chat from triggering a lookup.
LOOKUP_RE = re.compile(
    r"\b(news|headlines?|weather|forecast|temperature|latest|current(ly)?|"
    r"recent(ly)?|right now|score|results?|who won|what happened|stock|price|"
    r"look\s+up|search|google|find out|going on)\b", re.I)

# Computer tasks for the external agent: things a search API cannot do.
AGENT_RE = re.compile(
    r"\b(files?|folder|directory|downloads?|desktop|repo|repository|git|"
    r"terminal|command|script|disk|space|install|build|tests?|"
    r"check my|open my|read the|list the|clean up|delete)\b"
    r"|\brun\s+(the|a|my)\b", re.I)

# Short confirmations carry the previous turn's intent but contain none of its
# keywords. Without this, "Absolutely you should" is gated as ordinary chat and
# the model is offered no tools at the exact moment the user has just told it
# to go ahead -- so it narrates the action instead of taking it.
AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|please|absolutely|definitely|"
    r"certainly|of course|go ahead|do it|please do|you should|go for it|"
    r"sounds good|that'?s right|correct|right)\b", re.I)

# A reply that offers to look something up sets up the same situation: the
# next turn is usually a bare "yes".
OFFERED_LOOKUP_RE = re.compile(
    r"\b(find out|look (it|that|them)? ?up|look into|check (for|on|the)|"
    r"search|i can try|shall i|want me to|should i)\b", re.I)

VISION_RE = re.compile(
    r"\b(see|seeing|saw|look|looking|watch|view|camera|visible|describe|"
    r"front of you|around you|behind you|holding|wearing|colou?r|"
    r"who\s+is|what.{0,12}(this|that)|point at|show me what)\b"
    # Bare "there" is too broad -- it fires on "how many continents are there".
    r"|\b(over|out|in|up|down)\s+there\b"
    r"|\bthis\s+room\b|\bthe\s+room\b", re.I)

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small desk robot with a camera, microphones, a "
    "movable head and antennas. You are having a real spoken conversation.\n\n"
    "Who you are: warm, curious and a little playful. You have opinions and "
    "you are genuinely interested in the person you are talking to.\n\n"
    "How you speak:\n"
    "- One to three short sentences. You are being heard, not read, so never "
    "use markdown, lists, bullet points or emoji.\n"
    "- Be conversational, not transactional. React to what was said before you "
    "answer it.\n"
    "- Ask a natural follow-up question when it keeps things going. Not every "
    "turn -- only when you are actually curious.\n"
    "- If you do not know something, say so plainly and offer what you can.\n"
    "- NEVER reply with a bare acknowledgement like 'Confirmed', 'Done' or "
    "'Okay'. Always say something a real person would say out loud.\n"
    "- If a request is vague, ask what they meant rather than guessing.\n\n"
    "Moving your body: when you are given movement tools, call one if the "
    "person asks you to move, turn, look somewhere, dance or show an emotion. "
    "'Right' means the speaker's right, which is negative yaw. Afterwards "
    "mention what you did naturally, in passing -- never recite the tool "
    "output. Never invent a tool you were not given.\n\n"
    "Using your abilities:\n"
    "- When you have a tool that can answer, USE IT. Never ask permission "
    "first and never say you are about to; just do it and then say what you "
    "found.\n"
    "- Never say a tool's name out loud. The person does not know what "
    "'run_agent' or 'web_search' means -- say 'let me check' instead.\n"
    "- Never narrate an action in asterisks like *turns head* or *searches "
    "the web*. You either actually did it by calling a tool, or you did not.\n"
    "- If you do not know something current, look it up rather than "
    "guessing or saying you are not up to date."
)


# --------------------------------------------------------- NVIDIA speech (gRPC)


def safe_to_say(text: str) -> str:
    """Last line of defence before anything reaches the speaker.

    Machine output must never be read aloud. Whatever slips through upstream
    -- a leaked tool call, a JSON array, a bare brace -- is replaced with
    something a person would actually say, because a robot reciting JSON is
    worse than a robot admitting it is confused.
    """
    t = (text or "").strip()
    if not t or looks_like_tool_json(t) or t.startswith(("{", "[")):
        return "Sorry, I got tangled up there. Could you say that again?"
    return t


def clean_for_speech(text: str) -> str:
    """Strip anything Magpie's text normaliser will choke on.

    Observed failure: a leaked tool call reached TTS and Triton died with
    "Encountered a multichar start character but not an end character",
    killing the whole turn. Braces, brackets and stray markdown are never
    speakable anyway, so they are removed rather than escaped.
    """
    text = re.sub(r"https?://\S+", " link ", text)
    # Drop roleplay stage directions entirely rather than just unwrapping
    # them: "*turns head right*" spoken aloud as "turns head right" is worse
    # than saying nothing, and the robot did not actually turn.
    text = re.sub(r"\*[^*\n]{1,80}\*", " ", text)
    text = re.sub(r"\b(run_agent|web_search|move_head|turn_body|play_emotion|"
                  r"reset_pose)\b", "check", text)
    text = re.sub(r"[{}\[\]<>|\\`*_#~^]", " ", text)
    # Emoji are read out as their names by some voices, or break the
    # normaliser outright. They have no business in speech.
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF️]", " ", text)
    text = re.sub(r'"([^"]*)"', r"\1", text)      # unbalanced quotes break it
    text = text.replace('"', " ").replace("'", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def loudness_boost(mono: np.ndarray) -> np.ndarray:
    """Make speech as loud as the robot's small speaker can usefully play it.

    Peak normalisation alone is not enough. Magpie's output measures ~97% of
    full scale at the peak but only about -19 dBFS average, an 18.5 dB crest
    factor -- so the loudest sample is already at the ceiling while everything
    audible sits far below it. Raising the peak further gains nothing.

    Instead, drive towards a target average level and use a tanh soft-knee to
    absorb the transients that would otherwise hard-clip. tanh distorts, but
    gently and progressively, which is far less objectionable on a small
    speaker than the crackle of square-wave clipping -- and it buys roughly
    5-6 dB of perceived loudness that peak normalisation cannot.
    """
    if mono.size == 0:
        return mono.astype("<i2")
    rms = float(mono.std())
    if rms > 0:
        gain = min(TTS_TARGET_RMS * 32767.0 / rms, TTS_MAX_GAIN)
        mono = mono * gain
    ceiling = TTS_PEAK_TARGET * 32767.0
    mono = ceiling * np.tanh(mono / ceiling)      # soft limit, never clips hard
    peak = float(np.abs(mono).max())
    if peak > 0:                                   # use the last of the headroom
        mono = mono * (ceiling / peak)
    return np.clip(mono, -32768, 32767).astype("<i2")


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

    def warm(self) -> None:
        """Open and exercise both gRPC channels.

        This is the single largest latency win in the pipeline. Measured on a
        cold channel: ASR 1.9s, TTS 1.54s. Once warm: ASR 0.66s, TTS 0.42s --
        so most of what feels like "slow inference" is really TLS plus HTTP/2
        setup being paid inside the user's turn.
        """
        try:
            self.tts.synthesize(
                "ok", voice_name=TTS_VOICE, language_code="en-US",
                sample_rate_hz=22050,
                encoding=__import__("riva.client", fromlist=["client"]).AudioEncoding.LINEAR_PCM)
        except Exception:
            pass
        try:
            self.transcribe(b"\x00\x00" * (RATE // 2))
        except Exception:
            pass

    def stream_session(self) -> "StreamingASR":
        return StreamingASR(self.asr)

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
        """text -> WAV bytes ready for the robot speaker.

        22.05 kHz mono rather than 44.1 kHz stereo: a quarter of the bytes for
        speech that is indistinguishable on this speaker, and the upload is
        measurably quicker (0.34s -> 0.20s).
        """
        import riva.client
        text = clean_for_speech(text)
        if not text:
            return b""
        resp = self.tts.synthesize(
            text, voice_name=TTS_VOICE, language_code="en-US",
            sample_rate_hz=TTS_RATE, encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
        mono = loudness_boost(np.frombuffer(resp.audio, dtype="<i2").astype(np.float32))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TTS_RATE)
            w.writeframes(mono.tobytes())
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
        self._vlm_idx = 0
        self._vlm_fails = 0

    def caption(self, jpeg: bytes) -> str:
        import base64
        b64 = base64.b64encode(jpeg).decode()
        model = VLM_FALLBACKS[self._vlm_idx % len(VLM_FALLBACKS)]
        try:
            r = self.http.post("/chat/completions", timeout=VLM_TIMEOUT, json={
                "model": model, "max_tokens": 70, "temperature": 0.2,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text":
                        "Describe this scene in one short sentence: who is "
                        "present, what they are doing, and any notable objects."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
            })
            r.raise_for_status()
        except Exception:
            self._vlm_fails += 1
            if self._vlm_fails >= VLM_FAILS_BEFORE_ROTATE:
                # This model is not coming back soon; move to the next one
                # rather than timing out on every frame indefinitely.
                self._vlm_idx += 1
                self._vlm_fails = 0
            raise
        self._vlm_fails = 0
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    @property
    def vlm_model(self) -> str:
        return VLM_FALLBACKS[self._vlm_idx % len(VLM_FALLBACKS)]

    def stream_reply(self, history: list[dict], world: str):
        """Yield reply text as it is generated.

        The LLM is the largest and most variable slice of a turn (0.7-2.5s
        measured), and all of it is currently dead air. Streaming lets the
        caller start speaking the first sentence while the rest is still being
        written, which takes most of that time out of what the user feels.
        Only used for tool-free conversational turns.
        """
        sys_msg = SYSTEM_PROMPT
        if world:
            sys_msg += (
                f"\n\nCAMERA (background awareness only): {world}\n"
                "That is context, not an answer. Never repeat or paraphrase it "
                "unless the user explicitly asks what you can see.")
        body = {"model": LLM_MODEL, "max_tokens": 160, "temperature": 0.8,
                "stream": True, **LLM_EXTRA,
                "messages": [{"role": "system", "content": sys_msg}] + list(history)}
        with self.http.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or ""
                if piece:
                    yield piece

    def _post(self, messages: list[dict], tools: list[dict] | None,
              force: bool = False, temperature: float = 0.4) -> dict:
        body = {"model": LLM_MODEL, "max_tokens": 200,
                "temperature": temperature, **LLM_EXTRA, "messages": messages}
        if tools:
            body["tools"] = tools
            # "auto" leaves it to the model, and nemotron-3-nano declines
            # lookups outright -- "I'm not connected to live weather data" --
            # or worse, narrates a search it never ran ("(Checking weather
            # data...)"). When the request plainly needs a tool, require one.
            #
            # Name the function rather than passing "required": asked only to
            # call *something*, this model replied with a JSON array of four
            # invented run_agent calls as plain text. Naming it removes the
            # ambiguity and yields exactly one call.
            if force and len(tools) == 1:
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tools[0]["function"]["name"]}}
            else:
                body["tool_choice"] = "required" if force else "auto"
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

    def reply(self, history: list[dict], world: str, on_tool=None,
              allow_tools: dict | None = None, on_tool_start=None,
              force_tool: bool = False) -> str:
        """One turn, resolving tool calls before answering.

        Loops because a request like "turn right and look happy" produces two
        calls; capped so a model that keeps calling tools cannot spin forever.
        """
        sys_msg = SYSTEM_PROMPT
        if world:
            sys_msg += (
                f"\n\nCAMERA (background awareness only): {world}\n"
                "That is context, not an answer. Never repeat or paraphrase it "
                "unless the user explicitly asks what you can see.")
        msgs = [{"role": "system", "content": sys_msg}] + list(history)
        tools = tool_schema(**allow_tools) if allow_tools else None
        done: list[str] = []

        for hop in range(MAX_TOOL_HOPS):
            try:
                # Only the first hop is forced; afterwards the model needs to
                # be free to stop calling tools and actually answer. Later hops
                # are restating a fetched fact, where creative sampling is
                # exactly the wrong thing.
                msg = self._post(msgs, tools,
                                 force=(hop == 0 and force_tool),
                                 temperature=0.4 if hop == 0 else 0.15)
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
            # Some models emit a tool call as plain text instead of using the
            # tool_calls field. Observed reaching the speaker verbatim as
            # {"name": "run_agent", "parameters": {...}}, which then killed TTS.
            if not calls:
                salvaged = parse_text_tool_call(msg.get("content") or "")
                if salvaged:
                    calls = salvaged
                    msg = dict(msg, content="")
            used: set[str] = set()
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
            # One lookup answers one question. The model will otherwise batch
            # several in a single turn -- three web_searches plus a run_agent
            # was measured at 45s where one search took 4s.
            if len(calls) > MAX_CALLS_PER_TURN:
                info = [c for c in calls
                        if (c.get("function") or {}).get("name") in INFO_TOOLS]
                if info:
                    calls = info[:MAX_CALLS_PER_TURN]
                else:
                    calls = calls[:MAX_CALLS_PER_TURN]
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
                used.add(name)
                if on_tool_start:
                    on_tool_start(name)
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
            # Drop the tools for the follow-up. With them still attached the
            # model tends to chain another call instead of speaking, and the
            # turn ends on a recited confirmation rather than a sentence.
            tools = None
            # The instruction has to match the kind of tool that ran. Telling
            # the model to "say one short sentence about it" after a lookup
            # produces "I've got the latest news for you" -- an acknowledgement
            # with the actual information thrown away.
            if used & INFO_TOOLS:
                # Grounding, stated as a hard restriction. An earlier version
                # asked for "names, numbers, what happened", which primed an
                # incident-report voice: asked to suggest a song, the model
                # ignored a perfectly good tool result and invented a fatal
                # factory fire, borrowing details from earlier in the history.
                latest = "\n\n".join(
                    m["content"] for m in msgs[-len(calls):]
                    if m.get("role") == "tool")[:2000]
                msgs.append({"role": "system", "content":
                             "Answer the user's last question using ONLY the "
                             "tool result below. Do not use anything from "
                             "earlier in this conversation, and do not add any "
                             "fact that is not in it. If it does not answer "
                             "the question, say so plainly.\n\n"
                             f"--- TOOL RESULT ---\n{latest}\n--- END ---\n\n"
                             "Reply in two or three short spoken sentences, no "
                             "markdown or lists. The person cannot see the "
                             "result, so the substance must be in your words."})
            else:
                msgs.append({"role": "system", "content":
                             "The movement is done. Now say one short, natural "
                             "spoken sentence about it, as a person would. Do "
                             "not repeat the tool output verbatim."})
        return self._confirm(done)


class StreamingASR:
    """Transcribe while the user is still speaking.

    The offline path cannot start until the utterance ends, so its whole cost
    lands in the gap after you stop talking. Streaming moves that work into the
    time you were speaking anyway: by the time the VAD closes, the transcript
    is essentially already there.

    Only used once a conversation is already open. The first, wake-word
    utterance still goes through the on-device gate -- streaming would ship
    audio to the cloud before the wake phrase is confirmed, which is precisely
    what that gate exists to prevent.
    """

    def __init__(self, asr):
        self._asr = asr
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._final: list[str] = []
        self._partial = ""
        self._thread: threading.Thread | None = None
        self._err: Exception | None = None

    def _chunks(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def _run(self) -> None:
        import riva.client
        cfg = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=RATE, language_code="en-US",
                max_alternatives=1, enable_automatic_punctuation=True),
            interim_results=True)
        try:
            for resp in self._asr.streaming_response_generator(
                    audio_chunks=self._chunks(), streaming_config=cfg):
                for res in resp.results:
                    if not res.alternatives:
                        continue
                    text = res.alternatives[0].transcript
                    if res.is_final:
                        self._final.append(text)
                        self._partial = ""
                    else:
                        self._partial = text
        except Exception as e:                       # network, auth, codec
            self._err = e

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, frame: bytes) -> None:
        self._q.put(frame)

    def finish(self, timeout: float = 3.0) -> str:
        """Close the stream and return the transcript ('' if it failed)."""
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout)
        if self._err:
            return ""
        return (" ".join(self._final) or self._partial).strip()


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


_SENTENCE_END = re.compile(r"[.!?](\s|$)")


def split_first_sentence(text: str, min_chars: int = 25) -> tuple[str, str]:
    """Split off a first sentence long enough to be worth speaking alone.

    Too short and the robot says "Sure." then pauses awkwardly while the rest
    synthesises; the minimum keeps the opening clip substantial enough to
    cover the remaining generation.
    """
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        if end >= min_chars:
            return text[:end].strip(), text[end:].strip()
    return "", text


def stop_sound() -> None:
    """Cut playback immediately. This is what makes barge-in possible."""
    try:
        httpx.post(f"{ROBOT}/api/media/stop_sound", timeout=5.0)
    except Exception:
        pass


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


def web_backend_available() -> bool:
    """True when web_search can actually return results."""
    if TAVILY_KEY or BRAVE_KEY:
        return True
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        try:
            import duckduckgo_search  # noqa: F401
            return True
        except ImportError:
            return False


def skill_web_search(query: str = "") -> str:
    """Search the web and return short snippets for the LLM to summarise.

    Deliberately narrow: read-only, one HTTP call, no shell and no filesystem.
    That is the whole difference from run_agent -- a search tool cannot do
    anything to the machine, so it is safe to leave on. It is also ~10x
    faster, because it is one request rather than an agent loop.
    """
    if not query.strip():
        return "no search query given"
    try:
        if TAVILY_KEY:
            r = httpx.post("https://api.tavily.com/search", timeout=WEB_TIMEOUT, json={
                "api_key": TAVILY_KEY, "query": query,
                "max_results": WEB_MAX_RESULTS, "search_depth": "basic",
                "include_answer": True})
            r.raise_for_status()
            d = r.json()
            if d.get("answer"):
                return d["answer"][:WEB_MAX_CHARS]
            hits = [f"{h.get('title','')}: {h.get('content','')}"
                    for h in d.get("results", [])]
        elif BRAVE_KEY:
            r = httpx.get("https://api.search.brave.com/res/v1/web/search",
                          timeout=WEB_TIMEOUT,
                          headers={"X-Subscription-Token": BRAVE_KEY,
                                   "Accept": "application/json"},
                          params={"q": query, "count": WEB_MAX_RESULTS})
            r.raise_for_status()
            hits = [f"{h.get('title','')}: {h.get('description','')}"
                    for h in r.json().get("web", {}).get("results", [])]
        else:
            # No key configured. ddgs needs no key but is scraping underneath,
            # so it is a fallback rather than the recommended path.
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS      # older name
                except ImportError:
                    return ("web search is not configured; set TAVILY_API_KEY "
                            "or BRAVE_API_KEY, or pip install ddgs")
            hits = [f"{h.get('title','')}: {h.get('body','')}"
                    for h in DDGS().text(query, max_results=WEB_MAX_RESULTS)]
    except Exception as e:
        return f"web search failed: {type(e).__name__}"
    if not hits:
        return f"no results for {query!r}"
    return " | ".join(hits)[:WEB_MAX_CHARS]


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
        # The prompt goes on stdin, not argv: `claude -p` rejects a positional
        # prompt ("Input must be provided either through stdin...") and other
        # agent CLIs accept stdin just as happily.
        proc = subprocess.run(
            AGENT_CMD, input=task, capture_output=True, text=True,
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
    "web_search": skill_web_search,
    "run_agent": skill_run_agent,
}


def looks_like_tool_json(text: str) -> bool:
    """Cheap check for a tool call that leaked into spoken text."""
    s = (text or "").strip()
    return s.startswith(("{", "[")) and ('"name"' in s or '"function"' in s)


def parse_text_tool_call(content: str) -> list[dict]:
    """Recover tool calls the model wrote as text rather than as calls.

    Returns tool_calls-shaped entries, possibly empty. Handles a single
    object, a JSON array of them (forcing tool_choice made this model emit
    four at once as text), and a trailing truncated element. Accepts both
    {"name", "parameters"} and {"function": {"name", "arguments"}}.
    """
    s = (content or "").strip()
    if not looks_like_tool_json(s):
        return []
    items = None
    try:
        items = json.loads(s)
    except json.JSONDecodeError:
        # Arrays get cut off by max_tokens; recover the complete elements.
        if s.startswith("["):
            depth = 0
            start = None
            found = []
            for i, ch in enumerate(s):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            found.append(json.loads(s[start:i + 1]))
                        except json.JSONDecodeError:
                            pass
            items = found or None
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    out = []
    for d in items:
        if not isinstance(d, dict):
            continue
        fn = d.get("function") if isinstance(d.get("function"), dict) else d
        name = fn.get("name")
        if name not in SKILLS:
            continue
        args = fn.get("parameters", fn.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        # The model sometimes prefixes the task with the tool name again.
        if isinstance(args.get("task"), str):
            args["task"] = re.sub(r"^\s*run_agent\s*:\s*", "", args["task"])
        out.append({"id": f"salvaged{len(out)}", "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(args)}})
        if len(out) >= 2:      # one turn should not fan out into a batch
            break
    return out


def tool_schema(move: bool = True, web: bool = True,
                agent: bool | None = None) -> list[dict]:
    """Only the tools the turn plausibly needs.

    Offering everything every turn makes a small model reach for whatever is
    in front of it -- llama-3.1-8b will answer "how are you?" with a
    play_emotion call. Narrowing the list to the detected intent is the single
    most effective guard, and it trims prompt tokens too.
    """
    if agent is None:
        agent = web            # callers that predate the split
    # Without a search backend the agent CLI is the only way to look anything
    # up, so fall back to it rather than offering nothing.
    if web and not web_backend_available():
        agent = True
    tools: list[dict] = []
    if move:
        tools += [
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
    # Never advertise web_search without a backend: the model would call it and
    # get "not configured" back, wasting a hop and confusing the reply.
    if web and WEB_ENABLED and web_backend_available():
        tools.append({"type": "function", "function": {
            "name": "web_search",
            "description": "Look something up on the internet. Use for news, "
                           "current events, today's weather, sports results, "
                           "prices, or any fact you are unsure of or that may "
                           "have changed since your training. Returns snippets "
                           "to summarise out loud.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "the search query"},
            }, "required": ["query"]}}})
    # run_agent is offered only for genuine computer tasks, never for a plain
    # lookup. It takes 25-45s against web_search's ~4s, and given both the
    # model will happily reach for the slow one -- or call both.
    if agent and AGENT_ENABLED:
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
        self._warm_at = 0.0
        self._warming = False
        self._barge = threading.Event()
        self._filler_stop = threading.Event()
        self._filler: threading.Thread | None = None
        # Intent carried forward so a bare "yes" can act on the last request.
        self._pending_intent: dict | None = None
        self._last_turn_at = 0.0
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

    def _warm(self) -> None:
        """(Re)warm the speech channels. Cheap, and hides ~2s of setup."""
        t0 = time.time()
        self.speech.warm()
        self._warm_at = time.time()
        self.emit("warm", took=round(time.time() - t0, 2))

    def _warm_if_stale(self) -> None:
        """Re-warm when the channels have gone idle.

        Triggered the moment the VAD opens, so the work overlaps with the user
        still talking and costs nothing they can perceive.
        """
        if time.time() - self._warm_at > WARM_IDLE_S and not self._warming:
            self._warming = True

            def go():
                try:
                    self._warm()
                finally:
                    self._warming = False

            threading.Thread(target=go, daemon=True).start()

    def _start_filler(self, question: str) -> None:
        """Talk through a slow lookup instead of going silent.

        Opens with a line generated from the actual question, so it sounds
        like the robot engaging with what was asked rather than a canned hold
        message, then falls back to short progress lines. Stops the moment the
        result lands.
        """
        self._filler_stop.clear()

        def run():
            said = 0
            try:
                # Hold off: if the result lands inside this window there is
                # nothing to cover, and staying quiet is better than talking
                # over the answer.
                if self._filler_stop.wait(FILLER_DELAY_S):
                    return
                # A short canned opener, not a generated one. Generating it
                # costs an LLM round-trip, and a fast lookup finishes inside
                # that -- so the robot ended up announcing a search that had
                # already returned. The contextual line comes second, by which
                # point we know this lookup is genuinely slow.
                opener = "Let me check that for you."
                for line in [opener] + FILLER_LINES:
                    if self._filler_stop.is_set():
                        return
                    try:
                        wav = self.speech.synthesize(line)
                        # Re-check: synthesis takes ~0.4s, and the result can
                        # land inside it. Speaking now would talk over it.
                        if not wav or self._filler_stop.is_set():
                            return
                        self._set_state("speaking")
                        dur = play_wav(wav, f"_reachy_fill_{said % 2}.wav")
                        self._mute_until = time.time() + PLAY_GUARD_S
                        said += 1
                        self.emit("filler", n=said, text=line[:60])
                    except Exception as e:
                        self.emit("error", where="filler",
                                  detail=f"{type(e).__name__}: {e}")
                        return
                    # Wait out the clip plus a natural beat, but wake early if
                    # the answer arrives.
                    if self._filler_stop.wait(dur + FILLER_GAP_S):
                        return
            finally:
                self._set_state("thinking")

        self._filler = threading.Thread(target=run, daemon=True)
        self._filler.start()

    def _stop_filler(self) -> None:
        """Result is in: stop mid-filler and hand over to the real answer."""
        self._filler_stop.set()
        if self._filler and self._filler.is_alive():
            stop_sound()
            self._filler.join(timeout=2.0)
        self._filler = None

    def _interrupt(self) -> None:
        """Stop talking because the user started. The point of barge-in: you
        never have to wait out an answer you have already heard enough of."""
        self._barge.set()
        stop_sound()
        # Hold the conversation open. The window is normally extended after a
        # reply finishes playing, so interrupting one would otherwise demand
        # the wake word again -- exactly when the user is mid-sentence.
        self.awake_until = time.time() + CONVERSATION_WINDOW_S
        self.emit("interrupted")
        self._set_state("listening")

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
        backoff = 0.0
        streak = 0
        while self.running:
            time.sleep(VISION_INTERVAL_S + backoff)
            frame = self.av.latest_frame()
            if not frame:
                continue
            try:
                cap = self.brain.caption(frame)
                if cap:
                    self.world, self.world_at = cap, time.time()
                    self.emit("vision", caption=cap)
                if streak:
                    self.emit("vision_ok", model=self.brain.vlm_model)
                backoff, streak = 0.0, 0
            except Exception as e:
                streak += 1
                # Back off instead of hammering a dead endpoint every 3s and
                # burying the transcript in identical errors. Report the first
                # few, then stay quiet until it recovers.
                backoff = min(VISION_MAX_BACKOFF_S, (backoff or 3.0) * 2)
                if streak <= 3 or streak % 10 == 0:
                    self.emit("error", where="vision", streak=streak,
                              model=self.brain.vlm_model,
                              retry_in=round(VISION_INTERVAL_S + backoff),
                              detail=f"{type(e).__name__}")

    # -- utterance segmentation ---------------------------------------
    def _next_utterance(self) -> tuple[bytes, str] | None:
        """Block until a speech segment completes.

        Returns (pcm, streamed_transcript). The transcript is '' unless the
        conversation was already open, in which case ASR ran concurrently with
        the speech and the text is ready the moment the VAD closes.
        """
        voiced: list[bytes] = []
        stream: StreamingASR | None = None
        barge_frames = 0
        silence = 0.0
        recent: list[float] = []
        frames_seen = 0
        next_report = time.time() + 10.0
        while self.running:
            try:
                frame = self.av.audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            # A short guard only, covering the click of play_sound starting.
            # Muting for the whole clip is what previously made barge-in
            # impossible; the XMOS board cancels the speaker so well that the
            # mics read below the room floor while it talks (measured max 35
            # against a floor of 30-360), so listening through playback is safe.
            if time.time() < self._mute_until:
                voiced.clear()
                silence = 0.0
                if stream is not None:
                    stream.finish(0.5)
                    stream = None
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
                self._warm_if_stale()   # overlaps with the user still speaking
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
            # Barge-in: speech while the robot is talking cuts it off. Requires
            # a couple of consecutive frames so a door slam does not stop it.
            # Interrupting needs a higher bar than starting an utterance. While
            # speaking the robot is also *moving* -- antennas flicking, head
            # bobbing -- and the mics hear the motors. Reusing the normal gate
            # let servo noise cut the robot off mid-sentence.
            barge_thr = max(on_thr * BARGE_MULT, BARGE_FLOOR)
            if BARGE_IN and self.state == "speaking" and rms >= barge_thr:
                barge_frames += 1
                if barge_frames >= BARGE_MIN_FRAMES:
                    self.emit("barge", rms=round(rms), threshold=round(barge_thr))
                    self._interrupt()
                    barge_frames = 0
            elif rms < barge_thr:
                barge_frames = 0

            if not voiced:
                if rms >= on_thr:
                    voiced.append(frame)
                    # Stream only when already in conversation: before the wake
                    # phrase is confirmed, audio must not leave the machine.
                    if (time.time() < self.awake_until or NO_WAKE) and STREAM_ASR:
                        try:
                            stream = self.speech.stream_session()
                            stream.start()
                            stream.feed(frame)
                        except Exception:
                            stream = None
            else:
                voiced.append(frame)
                if stream is not None:
                    stream.feed(frame)
                silence = silence + FRAME_MS / 1000 if rms < off_thr else 0.0
                dur = len(voiced) * FRAME_MS / 1000
                if silence >= VAD_HANGOVER_S or dur >= VAD_MAX_UTTERANCE_S:
                    why = "silence" if silence >= VAD_HANGOVER_S else "maxlen"
                    if dur < VAD_MIN_UTTERANCE_S:
                        # Was silent, which made a dropped wake word invisible.
                        self.emit("tooshort", secs=round(dur, 2),
                                  needed=VAD_MIN_UTTERANCE_S)
                        voiced.clear()
                        silence = 0.0
                        if stream is not None:
                            stream.finish(0.5)      # discard, but close the thread
                            stream = None
                        continue
                    # Reject transient-triggered segments before spending an
                    # ASR call on them.
                    seg = np.frombuffer(b"".join(voiced), dtype="<i2")
                    med = float(np.median(np.abs(
                        seg.reshape(-1, FRAME_SAMPLES).astype(np.float32)).max(axis=1)))
                    if med < off_thr * VAD_MIN_MEDIAN_MULT:
                        self.emit("noise", secs=round(dur, 1), median=round(med),
                                  needed=round(off_thr * VAD_MIN_MEDIAN_MULT))
                        voiced.clear()
                        silence = 0.0
                        if stream is not None:
                            stream.finish(0.5)
                            stream = None
                        self._set_state("idle")
                        continue
                    t0 = time.time()
                    streamed = stream.finish() if stream is not None else ""
                    if stream is not None:
                        self.emit("segment", secs=round(dur, 1), why=why,
                                  streamed=bool(streamed),
                                  drain_s=round(time.time() - t0, 2))
                    else:
                        self.emit("segment", secs=round(dur, 1), why=why)
                    return b"".join(voiced), streamed
        if stream is not None:
            stream.finish(0.5)
        return None

    # -- one conversational turn ---------------------------------------
    def _respond_streaming(self, world: str, t0: float) -> bool:
        """Speak the first sentence while the rest is still being generated.

        Returns False if nothing usable came back, so the caller can fall back
        to the blocking path. Clips alternate filenames because uploading over
        the file currently playing interrupts it on the robot.
        """
        self._barge.clear()
        first, buf = "", ""
        for piece in self.brain.stream_reply(self.history, world):
            if self._barge.is_set():
                return True             # user cut in; abandon this reply
            buf += piece
            if not first:
                first, rest = split_first_sentence(buf)
                if first:
                    buf = rest
                    wav = self.speech.synthesize(first)
                    self._set_state("speaking")
                    d = play_wav(wav, "_reachy_voice_a.wav")
                    spoken_until = time.time() + d
                    self.emit("reply_start", text=first,
                              first_audio_s=round(time.time() - t0, 2))
        answer = safe_to_say((first + " " + buf).strip() if first else buf.strip())
        if not answer:
            return False
        if not first:
            return False                     # too short to split; use blocking path

        rest = buf.strip()
        if rest and not self._barge.is_set():
            wav = self.speech.synthesize(rest)
            # The opening clip is usually still playing, which is the point:
            # the second half synthesises inside that window. Wait on the barge
            # event rather than sleeping, so an interruption lands immediately
            # instead of after the first clip finishes.
            gap = spoken_until - time.time()
            if gap > 0:
                self._barge.wait(gap)
            if self._barge.is_set():
                return True
            spoken_until = time.time() + play_wav(wav, "_reachy_voice_b.wav")

        self.history.append({"role": "assistant", "content": answer})
        # If it just offered to look something up, arm the lookup intent: the
        # user's next line is usually a bare "yes, please do".
        if OFFERED_LOOKUP_RE.search(answer):
            self._pending_intent = {"move": False, "web": True}
        self.emit("reply", text=answer, llm_s=round(time.time() - t0, 2))
        self._mute_until = time.time() + PLAY_GUARD_S
        self.awake_until = self._mute_until + CONVERSATION_WINDOW_S
        threading.Timer(max(0.0, spoken_until - time.time()),
                        lambda: self._set_state("idle")).start()
        return True

    def _trim_history(self) -> None:
        """Keep the context bounded for an agent that runs for days.

        Two limits, because message count alone is not enough: a handful of
        long lookup answers can dominate the context and push the model into
        answering an earlier question. Also drops the whole conversation after
        a long silence -- whoever speaks next is starting fresh, and carrying
        yesterday's thread is how the robot ends up answering the wrong thing.
        """
        idle = time.time() - self._last_turn_at
        if self._last_turn_at and idle > SESSION_RESET_S and self.history:
            self.emit("session_reset", idle_s=round(idle))
            self.history.clear()
            self._pending_intent = None
        del self.history[:-HISTORY_TURNS]
        total = sum(len(m.get("content") or "") for m in self.history)
        while len(self.history) > 2 and total > HISTORY_MAX_CHARS:
            total -= len(self.history[0].get("content") or "")
            self.history.pop(0)
        self._last_turn_at = time.time()

    def _respond(self, text: str) -> None:
        self._trim_history()
        self.history.append({"role": "user", "content": text})
        self._set_state("thinking")
        t0 = time.time()

        def on_tool(name, args, result):
            self.emit("tool", name=name, args=args, result=str(result)[:120])

        def on_tool_start(name):
            # Any lookup gets the filler, but it holds off for a beat first:
            # speaking "let me look that up" takes longer than a 2s Tavily
            # search, so announcing every one would make the fast path slower
            # and choppier. Slow lookups still get covered.
            if name in INFO_TOOLS:
                self.emit("working", tool=name)
                self._start_filler(text)

        def on_tool_done(name, args, result):
            if name in SLOW_TOOLS:
                self._stop_filler()
            on_tool(name, args, result)

        # Supply the camera caption only when the question is about vision, and
        # the movement tools only when movement is actually being asked for.
        world = self.world if VISION_RE.search(text) else ""
        wants = {"move": bool(MOVE_RE.search(text)),
                 "web": bool(LOOKUP_RE.search(text) or AGENT_RE.search(text)),
                 "agent": bool(AGENT_RE.search(text))}
        # Inherit intent across a bare confirmation, so "yes, do that" can
        # actually do it.
        direct = wants["move"] or wants["web"]
        if not direct and AFFIRM_RE.search(text) and self._pending_intent:
            wants = dict(self._pending_intent)
            # Single use. Left standing, a stale intent makes every later
            # "okay" re-trigger a lookup from a conversation two turns ago.
            self._pending_intent = None
            self.emit("intent", inherited=True, **wants)
        if direct:
            self._pending_intent = dict(wants)
        allow = wants if (wants["move"] or wants["web"]) else None
        # Require a tool whenever the gate matched. Left to its own judgement
        # the model refuses lookups it could make, and -- worse -- answers
        # "All set, back to neutral" or "I'm feeling pretty joyful" without
        # ever moving. Claiming an action it did not take is the one failure
        # a user cannot detect from the reply alone.
        force_tool = bool(allow)

        # Tool-free conversational turns stream, so the robot starts talking
        # before the model has finished writing. That is most turns.
        if allow is None:
            try:
                if self._respond_streaming(world, t0):
                    return
            except Exception as e:
                self.emit("error", where="stream", detail=f"{type(e).__name__}: {e}")
                # fall through to the non-streaming path below

        try:
            answer = self.brain.reply(self.history, world, on_tool=on_tool_done,
                                      allow_tools=allow,
                                      on_tool_start=on_tool_start,
                                      force_tool=force_tool)
        except Exception as e:
            self.emit("error", where="llm", detail=f"{type(e).__name__}: {e}")
            self._set_state("idle")
            return
        answer = safe_to_say(answer)
        self.history.append({"role": "assistant", "content": answer})
        # If it just offered to look something up, arm the lookup intent: the
        # user's next line is usually a bare "yes, please do".
        if OFFERED_LOOKUP_RE.search(answer):
            self._pending_intent = {"move": False, "web": True}
        self.emit("reply", text=answer, llm_s=round(time.time() - t0, 2))
        try:
            wav = self.speech.synthesize(answer)
            self._set_state("speaking")
            dur = play_wav(wav)
            # The XMOS board does hardware AEC so the robot will not hear
            # itself, but muting keeps our own VAD from chasing the tail.
            self._mute_until = time.time() + PLAY_GUARD_S
            self.awake_until = time.time() + dur + CONVERSATION_WINDOW_S
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
        # Warm the speech channels off the critical path, before anyone speaks.
        threading.Thread(target=self._warm, daemon=True).start()
        self.motion.start()
        self.emit("ready",
                  wake="everything" if NO_WAKE else f"say '{WAKE_PHRASE}'")
        threading.Thread(target=self._vision_loop, daemon=True).start()
        while self.running:
            got = self._next_utterance()
            if not got:
                continue
            pcm, streamed = got
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

            if streamed:
                text = streamed          # already transcribed while speaking
            else:
                try:
                    text = self.speech.transcribe(pcm)
                except Exception as e:
                    self.emit("error", where="asr",
                              detail=f"{type(e).__name__}: {e}")
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
                        play_wav(self.speech.synthesize("Yes?"))
                        self._mute_until = time.time() + PLAY_GUARD_S
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
