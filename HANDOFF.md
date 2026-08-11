# Project handoff — Reachy Mini voice agent

Everything needed to pick this up on another machine. Written 2026-08-11.

Claude Code sessions do not transfer between machines, so this file replaces
the conversation: it records what was built, what was measured, which
approaches were tried and rejected, and what is still open.

---

## 1. Current state

Working and merged to `main` (`39fe49d`). The robot holds a spoken
conversation, can be driven by voice, sees through its camera, and can look
things up on the internet.

| Capability | State |
|---|---|
| Wake-word conversation ("hey") | working |
| On-device wake detection (faster-whisper) | working |
| Vision — captions every 3 s into a rolling world state | working |
| Voice movement control — head, body, 81 emotions | working |
| Web / computer tasks via an agent CLI | working, ~27 s, off by default |
| Web search via a search API | built, **needs an API key** |
| Web UI — video, audio meter, transcript, power button | working |
| Robot power on/off from the UI | working |

**First sound after you stop speaking: ~1.3–1.6 s.**

---

## 2. Starting on a new machine

```bash
git clone https://github.com/nileshai/reachy-voice-agent.git
cd reachy-voice-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install gstreamer libnice-gstreamer      # or apt, see README

cp .env.example .env && $EDITOR .env          # NVIDIA_API_KEY, REACHY_IP
set -a && source .env && set +a
.venv/bin/uvicorn server:app --port 7788
```

Then open <http://localhost:7788/voice.html> and press **Start conversation**.

**The API key is not in this repo and must be carried separately.** Get one at
[build.nvidia.com](https://build.nvidia.com). The key used during development
was exposed in a chat transcript and should be treated as compromised —
rotate it rather than reusing it.

Two host-specific things that are easy to lose a day to:

- **macOS 15+ needs Local Network permission per application.** Without it
  every connection to the robot fails with `Errno 65: No route to host`,
  which is indistinguishable from the robot being offline. Grant it to your
  terminal in System Settings → Privacy & Security → Local Network.
- **A full-tunnel VPN produces the identical error** by capturing the route to
  `192.168.x.x`. Check with `route -n get <robot-ip> | grep interface`;
  anything other than your LAN interface means the VPN has it.

---

## 3. Architecture, and why it is shaped this way

```
one gst-launch process ──┬─▶ 16 kHz PCM ─▶ VAD ─▶ local wake ─▶ ASR ─┐
(single WebRTC peer)     └─▶ JPEG @ 0.5 fps ─▶ VLM caption ─┐        │
                                                            ▼        ▼
                                                 rolling world state + text
                                                            │
                                                    NVIDIA LLM ▼ (streamed)
                                                    Magpie TTS ─▶ speaker + motion
```

**One gstreamer process for both streams.** The robot advertises a *single*
WebRTC producer. Two consumers fight over it and one loses — this is why the
web UI re-serves the agent's own frames instead of opening its own connection,
and why the preflight fails if the Robot Eyes panel is already open. Both pads
must be drained; leaving one unlinked kills the pipeline in ~0.5 s with
`not-linked (-1)`.

**Intent gating decides what the model is offered.** Three regexes route each
utterance: `MOVE_RE` (movement tools), `LOOKUP_RE`/`AGENT_RE` (search tools),
`VISION_RE` (camera caption). Nothing is offered unless matched. This exists
because a small model reaches for whatever is in front of it — see §5.

**Speech and vision run on separate clocks.** Vision captions run on their own
timer into a rolling world state, so vision latency never lands inside a turn.

---

## 4. Measured numbers

Not estimates — all measured against this robot and this account.

| Stage | Cold | Warm |
|---|---|---|
| ASR (Parakeet 1.1b), offline | 1.90 s | 0.95 s |
| ASR, streaming (drain after last frame) | — | **0.20 s** |
| LLM (llama-3.1-8b) | — | 0.6–5.7 s, streamed |
| LLM → first audio (streaming) | — | **0.7–1.0 s** |
| TTS (Magpie) | 1.54 s | 0.42 s |
| TTS (Chatterbox) | — | 1.68 s — 4× slower |
| Upload + play (22 kHz mono) | — | 0.20 s |
| VAD hangover | — | 0.40 s |

Room acoustics here: silence ≈ 30 rms per 100 ms frame, speech 400–13000.
VAD thresholds are multiples of a rolling p20 noise floor, not constants.

---

## 5. Things already tried that did not work

Do not spend time re-deriving these.

- **`mistral-nemotron` for the main LLM.** Fastest first token (0.22 s) but
  returns `tool_calls: []` — it explains what it *would* do and the robot never
  moves. Also slower end-to-end (1.90 s vs 0.63 s).
- **`gpt-oss-120b`** nests tool arguments wrongly: `{"degrees": {"yaw_deg": -30}}`.
- **`nemotron-3-super-120b`** emits correct tool calls but takes 25–30 s and
  leaks its reasoning into the content.
- **`cosmos-reason2-8b`** is listed in `/v1/models` but returns 404 — not
  entitled on this account. Worth requesting; it is the physical-reasoning
  model and a better fit than a general VLM.
- **`nemotron-3-nano-omni`** silently ignores `input_audio` and is text-only on
  the hosted endpoint despite the name.
- **Speech over REST.** All `/v1/audio/*` paths 404. ASR and TTS are gRPC-only
  via `grpc.nvcf.nvidia.com:443` with a function-id header.
- **Web search on build.nvidia.com.** There is none. The "search" NIMs are
  protein-MSA, OCR and 3D-asset retrieval. Web access must come from outside.
- **The two `/spark/` blueprints** (live-vlm-webui, reachy-photo-booth) are DGX
  Spark playbooks — local inference only, not callable as hosted APIs. Good
  architecture references, not a shortcut.
- **Browser WebRTC direct to the robot.** Chrome hides the local IP behind an
  mDNS ICE candidate the robot cannot resolve, so media never flows. Hence the
  server-side gstreamer → MJPEG path. This is *not* caused by the VPN.

---

## 6. Bugs found the hard way

Each of these cost real time; the fixes are in the code with comments.

- **Short reads on a raw pipe.** `stdout.read(n)` returns partial frames;
  zero-padding them spliced silence through the audio and inflated throughput
  4×. Accumulate to exact frame boundaries.
- **Noise-floor estimator deadlock.** Sampling only non-speech frames means
  that once the VAD gate latches open the estimator stops updating, the floor
  drifts to a speech-level value, and no utterance can ever close. Sample all
  frames, take a low percentile.
- **The daemon resets speaker volume on restart** (observed dropping to 62).
  Restored on power-on.
- **Peak normalisation is not loudness.** TTS measured 97 % of full scale but
  −19 dBFS average — an 18.5 dB crest factor. A tanh soft-knee limiter is
  +14 dB perceived where more peak gain does nothing.
- **`claude -p` rejects a positional prompt**; it must go on stdin, and
  headless mode declines web search unless `--allowedTools` names it.
- **Tool results are echoed verbatim as speech.** Phrase them for the ear
  ("looking up"), not for a log ("yaw 0, pitch −25").
- **Post-tool instructions must match the tool type.** "Say one short sentence
  about it" after a *lookup* produces "I've got the latest news for you" with
  the actual information discarded.
- **`pgrep -f gst-launch-1.0` matches the shell running the check.** Use `-x`.

---

## 7. Open items

1. **Rotate the NVIDIA API key.** Highest priority.
2. **Web search backend.** `web_search` is written and wired but has no key.
   A Tavily key drops news/weather from ~27 s (agent CLI) to ~3 s. Set
   `TAVILY_API_KEY` and it activates automatically.
3. **`vision ReadTimeout`** appears occasionally — the VLM captioning call
   timing out. Harmless (the caption goes stale, the loop retries) but wants
   retry/backoff.
4. **`docs/ui.png`** — the README references a screenshot that does not exist.
5. **Latency floor is ~1.3 s**: hangover 0.4 + drain 0.2 + first token 0.3 +
   TTS 0.4. Going lower needs speculative endpointing — starting the LLM on a
   partial transcript — which risks answering the wrong question.
6. **NeMoClaw** was never identified. `aiq` (NeMo Agent Toolkit) and the
   `claude` CLI are on the dev machine; nothing named nemoclaw was found. If
   it is a CLI taking a prompt, `REACHY_AGENT_CMD` already covers it.

---

## 8. Safety notes worth keeping

- **`REACHY_NO_WAKE=1` answers everything it hears** and sends every utterance
  to cloud ASR. It exists for a first end-to-end test. It once transcribed a
  live work meeting and replied out loud into the room. Do not use it around
  other people.
- **`REACHY_AGENT_ENABLED=1` grants shell and filesystem access to anything
  sayable near the robot**, unauthenticated, with a wake word of "hey" — a
  podcast is a plausible trigger. Off by default and omitted from the tool
  list unless enabled.
- **The wake word is a regex over a transcript**, not a trained spotter.
  `hey` is convenient and fires on ordinary conversation; `hey reachy` is far
  more selective. `REACHY_WAKE_PHRASE` switches it.
- **No authentication on port 7788.** Do not expose it beyond your LAN.
