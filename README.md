# Reachy Mini — Live Voice & Vision Agent

A conversational agent for the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/)
robot. It hears through the robot's mics, sees through its camera, thinks with
an NVIDIA-hosted LLM, and answers through the robot's speaker — with the head
and antennas moving to match what it's doing.

All inference runs on [build.nvidia.com](https://build.nvidia.com). No GPU
required locally; a laptop on the same network as the robot is enough.

```
  one gst-launch process  ──┬─▶ 16 kHz PCM ─▶ VAD ─▶ local wake ─▶ Parakeet ASR ─┐
  (single WebRTC peer)      └─▶ JPEG @ 0.5 fps ─▶ VLM caption ─┐                 │
                                                               ▼                 ▼
                                                    rolling world state  +  transcript
                                                               │
                                                       NVIDIA LLM ▼
                                                       Magpie TTS ─▶ speaker + motion
```

<!-- Add a screenshot of the web UI here: docs/ui.png -->

---

## What it does

- **Wake-word conversation** — say "hey", ask a question, hear an answer.
- **Sees while it talks** — a vision model captions the camera every few
  seconds into a rolling "world state" the LLM reads as context, so it can
  answer *"what am I holding?"* without paying vision latency per turn.
- **On-device wake detection** — audio never leaves your machine until the
  wake phrase is actually heard.
- **Web UI** — live video, audio meter with a visible VAD threshold, running
  transcript, and start/stop.
- **Expressive motion** — head and antennas animate differently when idle,
  listening, thinking, and speaking.
- **Voice movement control** — "turn 30 degrees right", "look up", "show me a
  happy emotion", "do a little dance". The LLM calls real tools against the
  robot; 81 emotions are reachable by plain English name.
- **Optional agent delegation** — hand tasks to an external agent CLI (files,
  web, shell). Off by default; see the warning below.

### Measured latency

Round-trip on a home network, models chosen for speed:

| Stage | Model | Time |
|---|---|---|
| Vision caption | `nemotron-nano-12b-v2-vl` | 1.5 s (off the critical path) |
| ASR | `parakeet-ctc-1.1b` | 1.9 s |
| LLM | `mistral-nemotron` | **0.4 s** |
| TTS (streaming) | `magpie-tts-multilingual` | 1.4 s to first audio |

≈ 2–3 s from end of speech to first sound.

---

## Requirements

- **Reachy Mini** (wireless) reachable on your network, daemon running
- **Python 3.10+**
- **GStreamer** with the WebRTC plugins
- An **NVIDIA API key** from [build.nvidia.com](https://build.nvidia.com) —
  free tier is enough to evaluate

```bash
# macOS
brew install gstreamer libnice-gstreamer

# Debian / Ubuntu
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad gstreamer1.0-nice
```

Verify: `gst-inspect-1.0 webrtcsrc` should print plugin details.

---

## Setup

```bash
git clone https://github.com/nileshai/reachy-voice-agent.git
cd reachy-voice-agent

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
$EDITOR .env          # set NVIDIA_API_KEY and REACHY_IP
```

Then load the environment and run:

```bash
set -a && source .env && set +a
.venv/bin/python voice_agent.py
```

It prints a preflight checklist before starting. Every line must pass:

```
Reachy voice agent — preflight
  [PASS] NVIDIA_API_KEY         nvapi-0Ly...
  [PASS] riva client
  [PASS] local wake             faster-whisper tiny.en (audio stays local)
  [PASS] robot daemon           http://192.168.1.2:8000
  [PASS] webrtc producer        f66a5b9d-027d-41ce-94bd-07624841ca59
  [PASS] gstreamer
  [PASS] producer free          ok
```

Say **"hey, what do you see?"**

### With the web UI

```bash
.venv/bin/uvicorn server:app --port 7788
```

Open **http://localhost:7788/voice.html**, click **Start conversation**.
`server.py` also provides a general control panel at `/` — motion, emotions,
sounds, diagnostics.

---

## Configuration

Everything is environment variables; see [`.env.example`](.env.example) for the
full annotated list. The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `NVIDIA_API_KEY` | — | **Required.** |
| `REACHY_IP` | `192.168.1.2` | Robot address. |
| `REACHY_WAKE_PHRASE` | `hey` | `hey reachy` is far more selective. |
| `REACHY_LOCAL_WAKE` | `1` | `0` sends every utterance to cloud ASR. |
| `REACHY_NO_WAKE` | `0` | `1` answers everything — see the warning below. |
| `REACHY_LLM` | `mistralai/mistral-nemotron` | Any model on build.nvidia.com. |
| `REACHY_VAD_ON_FLOOR` | `400` | Raise in a noisy room. |

### Privacy

The wake word exists so the robot ignores conversation not directed at it.
Two settings weaken that, deliberately:

- **`REACHY_NO_WAKE=1`** replies out loud to *everything* it hears and sends
  every utterance to cloud ASR. It exists to get a first end-to-end test
  working. **Do not use it around other people.**
- **`REACHY_LOCAL_WAKE=0`** still requires the wake word to *reply*, but every
  utterance is transcribed in the cloud to check for it.

With defaults (`REACHY_LOCAL_WAKE=1`), speech that isn't addressed to the robot
is transcribed locally, matched, and discarded — nothing leaves the machine.

---

## How it works

### One GStreamer process for both streams

The robot advertises a **single WebRTC producer**. Two consumers fight over it
and one loses. So a single `gst-launch-1.0` takes both pads — audio to stdout
as raw PCM, video to rotating JPEGs — and the web UI re-serves *those frames*
rather than opening its own connection.

If the camera panel in the main control UI is open, it holds the producer and
the agent can't start. The preflight catches this.

Both pads must be drained. Leaving one unlinked kills the pipeline in about
half a second with `not-linked (-1)`.

### Adaptive voice activity detection

Absolute thresholds don't survive a change of room. The VAD tracks a rolling
20th-percentile noise floor across **all** recent frames and sets its gate as a
multiple of it.

Sampling only non-speech frames seems tidier but deadlocks: if the gate latches
open, the estimator stops updating, the floor drifts to a speech-level value,
and nothing can ever close the utterance. Measured here: silence ≈ 30 rms,
speech 400–13000.

The `[audio]` heartbeat prints the floor, the gate, and recent peaks every
10 s — that's the tuning signal.

### Cloud ASR mangles wake words

Parakeet reliably renders "Reachy" as **"Rich"**. Wake matching therefore
accepts a set of variants, and when the local gate has already confirmed the
phrase, a second cloud-side match is *not* required — demanding one rejects
valid wakes.

### Voice movement control

The LLM is given four robot tools — `move_head`, `turn_body`, `play_emotion`,
`reset_pose` — and calls them when you ask it to move. A request like "look
right and show a curious emotion" produces two calls in one turn.

Three things this needed that aren't obvious:

- **Model choice is a hard constraint.** `mistral-nemotron` is the fastest
  model here but returns `tool_calls: []` and explains what it *would* do — the
  robot never moves. `gpt-oss-120b` nests arguments wrongly
  (`{"degrees": {"yaw_deg": -30}}`). `llama-3.1-8b-instruct` is the default
  because it emits correct calls at usable latency.
- **Arguments arrive as strings.** Models emit `"5"` rather than `5` often
  enough that every numeric argument is coerced before use; unexpected and
  nested keys are dropped rather than passed through.
- **Tool results are phrased for speech.** The model routinely echoes a tool's
  return value verbatim as its spoken reply, so results read "looking up", not
  "head to yaw 0, pitch -25, roll 0".

Emotion names are resolved through a synonym map: the library has `cheerful1`,
not `happy`, and no amount of fuzzy spelling matching bridges that.

Pitch direction was verified on hardware rather than assumed — positive pitch
aims the camera **down**, confirmed by pointing the head at ±35° and asking the
vision model which way it was looking.

### Agent delegation (off by default)

`run_agent` hands a task to an external CLI, set by `REACHY_AGENT_CMD`
(default `claude -p`):

```bash
REACHY_AGENT_ENABLED=1 REACHY_AGENT_CMD="claude -p" .venv/bin/python voice_agent.py
```

> **Think before enabling this.** It turns anything sayable within earshot into
> a command on your machine, with no authentication and no confirmation step.
> Anyone who can speak near the robot can drive it — and with a wake word as
> loose as `hey`, so can a podcast. Enable it only when you are alone and know
> exactly what your agent CLI is permitted to do.

### Speech and vision on separate clocks

Vision captions run on their own timer into a rolling world state. The LLM
reads the latest caption as context, so vision latency never lands inside a
conversational turn.

---

## HTTP API

Mounted at `/api/voice` when `server.py` runs:

| Endpoint | Purpose |
|---|---|
| `POST /start`, `POST /stop` | Run the agent |
| `POST /wake` | Open a conversation window — no wake phrase needed |
| `GET /status` | State, audio level, VAD gate, world caption, models |
| `GET /events?since=<ts>` | Transcript and lifecycle events |
| `GET /stream` | MJPEG of the agent's frames |
| `POST /say` | Speak arbitrary text |

Embed it in your own app:

```python
from voice_agent import router as voice_router
app.include_router(voice_router)   # before any catch-all static mount
```

---

## Troubleshooting

**`No route to host` / `Errno 65`, but the robot pings fine**
macOS 15+ requires **Local Network** permission per application. Grant it to
your terminal in System Settings → Privacy & Security → Local Network, then
restart the terminal. A denied app gets `EHOSTUNREACH` — indistinguishable
from the robot being offline. A VPN in full-tunnel mode causes the same error
by capturing the route to `192.168.x.x`.

**`[FAIL] producer free`**
Something else holds the camera. Close the Robot Eyes panel, or:
```bash
pkill -x gst-launch-1.0
```

**It never hears me — `[audio]` peak stays below the gate**
Move closer, or lower `REACHY_VAD_ON_FLOOR`.

**`[segment] why='maxlen'` then `background noise, no speech`**
The gate is stuck open on room noise. Raise `REACHY_VAD_ON_FLOOR` and
`REACHY_VAD_OFF_FLOOR`.

**Transcript shows my speech but nothing happens**
No wake word. The UI marks these "NOT ADDRESSED TO REACHY". Start with the
wake phrase or click **Talk to Reachy**.

**It's too quiet**
Already normalised to 97% of full scale on top of the robot's own volume. Raise
`REACHY_TTS_PEAK` toward `1.0`, and check `POST /api/volume/set`.

**Model returns 404**
Listed in `/v1/models` but not enabled for your account — request access on
build.nvidia.com, or pick another model.

---

## Limitations

- **Not a streaming pipeline.** ASR runs on complete utterances, so there's no
  barge-in — you can't interrupt mid-answer.
- **Wake word is regex over a transcript**, not a trained keyword spotter.
  `hey` will false-fire in conversation; `hey reachy` is much better.
- **English only** as configured. Magpie and Parakeet both support more —
  change `language_code`.
- **Browser WebRTC can't talk to this robot directly.** Chrome hides the local
  IP behind an mDNS ICE candidate the robot can't resolve, so media never
  flows. Hence the server-side GStreamer → MJPEG path.
- **No authentication.** Don't expose port 7788 beyond your LAN.

---

## Credits

Built on [Pollen Robotics](https://www.pollen-robotics.com/) Reachy Mini and
NVIDIA NIM / Riva. Inspired by NVIDIA's
[Spark Reachy Photo Booth](https://github.com/NVIDIA/spark-reachy-photo-booth),
which runs a similar pipeline entirely locally on a DGX Spark.

## License

MIT — see [LICENSE](LICENSE).
