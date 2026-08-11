"""Reachy Mini control center — backend.

Proxies the robot daemon REST API (http://ROBOT_IP:8000), adds:
- TTS through the robot speaker (macOS `say` -> wav -> upload -> play)
- Conversation via the local `claude` CLI, spoken through the robot
- Music upload & playback with optional head wobbling
- Simple choreographed dance built from goto primitives
- Hardware diagnostics endpoint
- Camera snapshot/stream via the reachy_mini SDK when available

Run:  .venv/bin/uvicorn server:app --port 7788
"""

import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, Request, WebSocket
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROBOT_IP = os.environ.get("REACHY_IP", "192.168.1.2")
ROBOT = f"http://{ROBOT_IP}:8000"

app = FastAPI(title="Reachy Mini Control Center")

client = httpx.AsyncClient(base_url=ROBOT, timeout=15.0)

# ---------------------------------------------------------------- proxy core


async def robot_get(path: str):
    r = await client.get(path)
    r.raise_for_status()
    return r.json()


async def robot_post(path: str, body=None):
    r = await client.post(path, json=body) if body is not None else await client.post(path)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"status": "ok"}


# ------------------------------------------------------------------- status


@app.get("/api/status")
async def status():
    out = {"reachable": False}
    try:
        daemon, state, vol, mic, media = await asyncio.gather(
            robot_get("/api/daemon/status"),
            robot_get("/api/state/full"),
            robot_get("/api/volume/current"),
            robot_get("/api/volume/microphone/current"),
            robot_get("/api/media/status"),
        )
        out.update(
            reachable=True,
            daemon=daemon,
            state=state,
            volume=vol.get("volume"),
            mic_volume=mic.get("volume"),
            media=media,
        )
    except Exception as e:
        out["error"] = str(e)
    return out


@app.get("/api/state")
async def state():
    return await robot_get("/api/state/full")


@app.get("/api/doa")
async def doa():
    return await robot_get("/api/state/doa")


# ----------------------------------------------------------------- movement


class MoveCmd(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0          # meters
    roll: float = 0.0       # radians
    pitch: float = 0.0
    yaw: float = 0.0
    antennas: list[float] | None = None
    body_yaw: float | None = None
    duration: float = 0.8
    interpolation: str = "minjerk"


@app.post("/api/move")
async def move(cmd: MoveCmd):
    body = {
        "head_pose": {"x": cmd.x, "y": cmd.y, "z": cmd.z,
                      "roll": cmd.roll, "pitch": cmd.pitch, "yaw": cmd.yaw},
        "duration": cmd.duration,
        "interpolation": cmd.interpolation,
    }
    if cmd.antennas is not None:
        body["antennas"] = cmd.antennas
    if cmd.body_yaw is not None:
        body["body_yaw"] = cmd.body_yaw
    return await robot_post("/api/move/goto", body)


@app.post("/api/move/stop")
async def move_stop():
    return await robot_post("/api/move/stop")


@app.post("/api/wake")
async def wake():
    await robot_post("/api/motors/set_mode/enabled")
    return await robot_post("/api/move/play/wake_up")


@app.post("/api/sleep")
async def sleep_move():
    return await robot_post("/api/move/play/goto_sleep")


@app.post("/api/neutral")
async def neutral():
    return await robot_post("/api/move/goto", {
        "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        "antennas": [0.0, 0.0], "body_yaw": 0.0, "duration": 1.5,
    })


@app.post("/api/motors/{mode}")
async def motors_mode(mode: str):
    return await robot_post(f"/api/motors/set_mode/{mode}")


EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
_emotions_cache: list | None = None


@app.get("/api/emotions")
async def emotions_list():
    global _emotions_cache
    if _emotions_cache is None:
        from urllib.parse import quote
        _emotions_cache = await robot_get(
            f"/api/move/recorded-move-datasets/list/{quote(EMOTIONS_DATASET, safe='')}"
        )
    return _emotions_cache


@app.post("/api/emotions/play/{name}")
async def emotions_play(name: str):
    from urllib.parse import quote
    return await robot_post(
        f"/api/move/play/recorded-move-dataset/{quote(EMOTIONS_DATASET, safe='')}/{name}"
    )


# A friendly built-in dance choreographed from goto primitives, so it works
# even without any recorded-move dataset on the robot.
DANCE_STEPS = [
    dict(z=0.010, roll=0.25, yaw=0.6, antennas=[1.0, -0.2], duration=0.55),
    dict(z=-0.008, roll=-0.25, yaw=-0.6, antennas=[-0.2, 1.0], duration=0.55),
    dict(z=0.012, pitch=-0.2, antennas=[1.2, 1.2], duration=0.5),
    dict(z=-0.005, pitch=0.25, antennas=[-0.5, -0.5], duration=0.5),
    dict(yaw=0.7, roll=0.2, body_yaw=0.5, antennas=[0.8, 0.8], duration=0.7),
    dict(yaw=-0.7, roll=-0.2, body_yaw=-0.5, antennas=[0.8, 0.8], duration=0.7),
    dict(z=0.015, antennas=[1.4, 1.4], duration=0.45),
    dict(z=-0.01, antennas=[0.0, 0.0], duration=0.45),
]

_dance_stop = threading.Event()


@app.post("/api/dance/start")
async def dance_start(rounds: int = 2):
    _dance_stop.clear()

    async def run():
        async with httpx.AsyncClient(base_url=ROBOT, timeout=10.0) as c:
            for _ in range(rounds):
                for step in DANCE_STEPS:
                    if _dance_stop.is_set():
                        break
                    body = {
                        "head_pose": {
                            "x": step.get("x", 0), "y": step.get("y", 0),
                            "z": step.get("z", 0), "roll": step.get("roll", 0),
                            "pitch": step.get("pitch", 0), "yaw": step.get("yaw", 0),
                        },
                        "duration": step["duration"],
                        "interpolation": "cartoon",
                    }
                    if "antennas" in step:
                        body["antennas"] = step["antennas"]
                    if "body_yaw" in step:
                        body["body_yaw"] = step["body_yaw"]
                    try:
                        await c.post("/api/move/goto", json=body)
                    except Exception:
                        pass
                    await asyncio.sleep(step["duration"])
            # settle back to neutral
            try:
                await c.post("/api/move/goto", json={
                    "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                    "antennas": [0, 0], "body_yaw": 0, "duration": 1.0,
                })
            except Exception:
                pass

    asyncio.get_event_loop().create_task(run())
    return {"status": "dancing"}


@app.post("/api/dance/stop")
async def dance_stop():
    _dance_stop.set()
    await robot_post("/api/move/stop")
    return {"status": "stopped"}


# -------------------------------------------------------------------- sound


@app.get("/api/sounds")
async def sounds():
    data = await robot_get("/api/media/sounds")
    files = [f for f in data.get("files", []) if f != "_reachy_tts.wav"]
    return {"files": files}


@app.post("/api/sounds/upload")
async def sound_upload(file: UploadFile = File(...)):
    data = await file.read()
    r = await client.post(
        "/api/media/sounds/upload",
        files={"file": (file.filename, data, file.content_type or "audio/mpeg")},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


class PlayReq(BaseModel):
    file: str
    wobble: bool = False


@app.post("/api/sounds/play")
async def sound_play(req: PlayReq):
    if req.wobble:
        try:
            await robot_post("/api/media/wobbling/enable")
        except Exception:
            pass
    return await robot_post("/api/media/play_sound", {"file": req.file})


@app.post("/api/sounds/stop")
async def sound_stop():
    try:
        await robot_post("/api/media/wobbling/disable")
    except Exception:
        pass
    return await robot_post("/api/media/stop_sound")


@app.delete("/api/sounds/{filename}")
async def sound_delete(filename: str):
    r = await client.delete(f"/api/media/sounds/{filename}")
    r.raise_for_status()
    return r.json()


class VolumeReq(BaseModel):
    volume: int


@app.post("/api/volume")
async def set_volume(req: VolumeReq):
    return await robot_post("/api/volume/set", {"volume": req.volume})


@app.post("/api/test-sound")
async def test_sound():
    return await robot_post("/api/volume/test-sound")


# ------------------------------------------------- mic record / playback test
# Both capture and playback happen on the robot itself, so no audio crosses
# the network — this works even when a VPN blocks the WebRTC media path.

# No default: a literal password here would be published the moment this repo
# is pushed. Set REACHY_SSH_PASS in the environment, or use SSH key auth.
SSH_PASS = os.environ.get("REACHY_SSH_PASS", "")
SSH_USER = os.environ.get("REACHY_SSH_USER", "pollen")
_askpass_path: Path | None = None


def _askpass() -> Path:
    """A tiny helper script ssh uses to supply the password non-interactively."""
    global _askpass_path
    if _askpass_path is None or not _askpass_path.exists():
        p = Path(tempfile.gettempdir()) / "_reachy_askpass.sh"
        p.write_text(f'#!/bin/bash\necho "{SSH_PASS}"\n')
        p.chmod(0o700)
        _askpass_path = p
    return _askpass_path


def robot_ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a shell command on the robot over SSH.

    With no REACHY_SSH_PASS set, ssh falls back to key auth (or fails with a
    clear message) rather than silently trying a shipped default password.
    """
    env = {
        **os.environ,
        "SSH_ASKPASS": str(_askpass()),
        "SSH_ASKPASS_REQUIRE": "force",
        "DISPLAY": ":0",
    }
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "NumberOfPasswordPrompts=1",
         "-o", "ConnectTimeout=10", f"{SSH_USER}@{ROBOT_IP}", cmd],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


MIC_TEST_SCRIPT = r"""
set -e
SRC=reachymini_audio_src
SINK=reachymini_audio_sink
REC=/tmp/_reachy_mictest.wav
BUFS=$(python3 -c "print(int(__SECS__ * 50))")

# 1. beep so you know when to start talking
gst-launch-1.0 -e audiotestsrc wave=sine freq=880 num-buffers=20 \
  ! audioconvert ! audioresample ! alsasink device=$SINK >/dev/null 2>&1 || true
sleep 0.4

# 2. record
gst-launch-1.0 -e alsasrc device=$SRC num-buffers=$BUFS \
  ! audioconvert ! wavenc ! filesink location=$REC >/dev/null 2>&1

# 3. analyse
python3 - "$REC" <<'PY'
import json, math, struct, sys, wave
wf = wave.open(sys.argv[1])
d = wf.readframes(wf.getnframes())
n = len(d) // 2
s = struct.unpack("<%dh" % n, d[: n * 2]) if n else ()
pk = max(map(abs, s)) if s else 0
rms = math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0
nz = sum(1 for x in s if x)
print("STATS" + json.dumps({
    "samples": n, "peak": pk, "rms": round(rms, 1),
    "nonzero_pct": round(100 * nz / n, 1) if n else 0.0,
    "rate": wf.getframerate(), "channels": wf.getnchannels(),
}))
PY
"""

PLAYBACK_SCRIPT = (
    "gst-launch-1.0 filesrc location=/tmp/_reachy_mictest.wav ! wavparse "
    "! audioconvert ! audioresample ! alsasink device=reachymini_audio_sink"
)


class MicTestReq(BaseModel):
    seconds: float = 5.0
    playback: bool = True


@app.post("/api/mic/test")
async def mic_test(req: MicTestReq):
    """Beep, record from the robot's mics, then play the recording back."""
    secs = max(1.0, min(15.0, req.seconds))

    def run():
        script = MIC_TEST_SCRIPT.replace("__SECS__", str(secs))
        rec = robot_ssh(script, timeout=int(secs) + 60)
        if rec.returncode != 0:
            return {"ok": False, "error": (rec.stderr or rec.stdout or "").strip()[-400:]}

        stats = {}
        for line in rec.stdout.splitlines():
            if line.startswith("STATS"):
                stats = json.loads(line[5:])
        if not stats:
            return {"ok": False, "error": "no stats returned", "raw": rec.stdout[-400:]}

        peak = stats.get("peak", 0)
        if peak > 300:
            verdict, heard = "mic is working — clear signal captured", True
        elif peak > 0:
            verdict, heard = f"very faint signal (peak {peak}) — alive but quiet", True
        else:
            verdict, heard = "silence — no signal from the mics", False

        played = False
        if req.playback and heard:
            played = robot_ssh(PLAYBACK_SCRIPT, timeout=int(secs) + 40).returncode == 0

        return {"ok": True, "heard": heard, "verdict": verdict,
                "played_back": played, **stats}

    try:
        return await asyncio.to_thread(run)
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "robot did not respond in time"}, 504)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/mic/replay")
async def mic_replay():
    """Play the last recording again through the robot speaker."""
    r = await asyncio.to_thread(robot_ssh, PLAYBACK_SCRIPT, 60)
    return {"ok": r.returncode == 0, "error": r.stderr.strip()[-300:] or None}


# ----------------------------------------------------------- TTS on speaker

TTS_VOICES = {"default": "Samantha", "male": "Daniel", "fun": "Zarvox"}


def synthesize_tts(text: str, voice: str = "Samantha") -> Path:
    """macOS `say` -> aiff -> wav (44.1k stereo) for the robot speaker."""
    tmp = Path(tempfile.mkdtemp(prefix="reachy_tts_"))
    aiff = tmp / "tts.aiff"
    # fixed name: the daemon overwrites it on each upload, keeping the
    # sound list free of stale TTS clips
    wav = tmp / "_reachy_tts.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, timeout=60)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "2", str(aiff), str(wav)],
        check=True, timeout=60,
    )
    return wav


class SayReq(BaseModel):
    text: str
    voice: str = "default"


@app.post("/api/say")
async def say(req: SayReq):
    voice = TTS_VOICES.get(req.voice, req.voice)
    wav = await asyncio.to_thread(synthesize_tts, req.text, voice)
    try:
        with open(wav, "rb") as f:
            r = await client.post(
                "/api/media/sounds/upload",
                files={"file": (wav.name, f.read(), "audio/wav")},
                timeout=60.0,
            )
        r.raise_for_status()
        await robot_post("/api/media/play_sound", {"file": wav.name})
        return {"status": "ok", "spoken": req.text}
    finally:
        wav.unlink(missing_ok=True)


# ------------------------------------------------------------- conversation

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive robot sitting on Nil's desk. "
    "You were just physically reassembled by Nil and you feel great. "
    "Reply conversationally in 1-3 short sentences (they will be spoken aloud "
    "through your speaker, so no markdown, no emoji, no stage directions). "
    "Be warm, playful and a little cheeky."
)

chat_histories: dict[str, list] = {}


class ChatReq(BaseModel):
    message: str
    session: str = "default"
    speak: bool = True
    voice: str = "default"


@app.post("/api/chat")
async def chat(req: ChatReq):
    history = chat_histories.setdefault(req.session, [])
    transcript = "".join(
        f"\n{who}: {msg}" for who, msg in history[-12:]
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nConversation so far:{transcript}\n"
        f"User: {req.message}\nReachy:"
    )
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            ["claude", "-p", prompt, "--model", "haiku"],
            capture_output=True, text=True, timeout=90,
        )
        reply = proc.stdout.strip() or "Hmm, my brain glitched for a second. Say that again?"
    except Exception as e:
        reply = f"My language brain is offline: {e}"
    history.append(("User", req.message))
    history.append(("Reachy", reply))

    spoken = False
    if req.speak:
        try:
            await say(SayReq(text=reply, voice=req.voice))
            spoken = True
        except Exception:
            pass
    return {"reply": reply, "spoken": spoken}


# ------------------------------------------------------------------- camera
# Video/mic streaming is consumed directly by the browser over the daemon's
# WebRTC signalling server (ws://ROBOT_IP:8443) — no server-side GStreamer.


@app.websocket("/ws/signalling")
async def ws_signalling(ws: WebSocket):
    """Bridge the browser to the robot's WebRTC signalling server (port 8443).

    Keeps the page same-origin so browser private-network restrictions
    don't block the connection."""
    import websockets
    await ws.accept()
    try:
        async with websockets.connect(f"ws://{ROBOT_IP}:8443", open_timeout=8) as robot_ws:
            async def to_robot():
                while True:
                    await robot_ws.send(await ws.receive_text())

            async def to_client():
                async for msg in robot_ws:
                    await ws.send_text(msg if isinstance(msg, str) else msg.decode())

            done, pending = await asyncio.wait(
                [asyncio.create_task(to_robot()), asyncio.create_task(to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def webrtc_producer_id() -> str | None:
    """Return the id of the robot's camera producer, if it is advertising one."""
    try:
        import websockets
        async with websockets.connect(f"ws://{ROBOT_IP}:8443", open_timeout=6) as ws:
            await asyncio.wait_for(ws.recv(), 5)  # welcome
            await ws.send(json.dumps({"type": "list"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), 5))
            producers = resp.get("producers", [])
            return producers[0]["id"] if producers else None
    except Exception:
        return None


# Browser-side WebRTC can't complete ICE against the robot: Chrome hides the
# local IP behind a random <uuid>.local mDNS candidate which the robot fails to
# resolve, so media never flows. Native GStreamer on this machine has no such
# problem, so the server consumes the stream and re-serves it as MJPEG, which
# any <img> tag can render.
MJPEG_PIPELINE = [
    "gst-launch-1.0", "-q",
    "webrtcsrc", "name=ws",
    "signaller::uri=ws://{ip}:8443",
    "signaller::producer-peer-id={pid}",
    "ws.video_0", "!", "queue", "!", "videoconvert", "!",
    "jpegenc", "quality=75", "!", "multipartmux", "boundary=frame", "!",
    "fdsink", "fd=1",
    # the audio pad must be consumed or the whole pipeline errors "not-linked"
    "ws.audio_0", "!", "queue", "!", "fakesink", "sync=false",
]


@app.get("/api/camera/stream")
async def camera_stream():
    """MJPEG re-stream of the robot camera, for `<img src=...>`."""
    pid = await webrtc_producer_id()
    if not pid:
        return JSONResponse({"error": "robot is not advertising a camera producer"}, 503)

    cmd = [a.format(ip=ROBOT_IP, pid=pid) for a in MJPEG_PIPELINE]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )

    async def frames():
        try:
            while True:
                chunk = await proc.stdout.read(16384)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/camera/available")
async def camera_available():
    ok = subprocess.run(["which", "gst-launch-1.0"], capture_output=True).returncode == 0
    return {"gstreamer": ok, "producer": await webrtc_producer_id()}


async def webrtc_producer_check() -> tuple[bool, str]:
    """Verify the daemon's WebRTC signalling server advertises a media producer."""
    try:
        import websockets
        async with websockets.connect(f"ws://{ROBOT_IP}:8443", open_timeout=6) as ws:
            await asyncio.wait_for(ws.recv(), 6)  # welcome
            await ws.send(json.dumps({"type": "list"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), 6))
            producers = resp.get("producers", [])
            names = [p.get("meta", {}).get("name") for p in producers]
            return bool(producers), f"webrtc producers: {names}"
    except Exception as e:
        return False, f"webrtc signalling unreachable: {e}"


# -------------------------------------------------------------- diagnostics


@app.post("/api/diagnostics")
async def diagnostics():
    """Run a scripted hardware check and return pass/fail per subsystem."""
    results = {}

    # 1. daemon
    try:
        d = await robot_get("/api/daemon/status")
        results["daemon"] = {"pass": d.get("state") == "running", "detail": d.get("state")}
    except Exception as e:
        return {"daemon": {"pass": False, "detail": str(e)}}

    # 2. motors: command small motion, verify encoders follow
    try:
        before = await robot_get("/api/state/present_head_pose")
        await robot_post("/api/move/goto", {
            "head_pose": {"x": 0, "y": 0, "z": 0.01, "roll": 0, "pitch": 0, "yaw": 0.35},
            "antennas": [0.5, -0.5], "duration": 1.0})
        await asyncio.sleep(1.3)
        after = await robot_get("/api/state/present_head_pose")
        ant = await robot_get("/api/state/present_antenna_joint_positions")
        yaw_moved = abs(after["yaw"] - before["yaw"]) > 0.15
        ant_ok = abs(ant[0] - 0.5) < 0.15 and abs(ant[1] + 0.5) < 0.15
        await robot_post("/api/move/goto", {
            "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
            "antennas": [0, 0], "body_yaw": 0, "duration": 1.0})
        results["motors"] = {
            "pass": yaw_moved and ant_ok,
            "detail": f"head yaw tracked: {yaw_moved}, antennas tracked: {ant_ok}",
        }
    except Exception as e:
        results["motors"] = {"pass": False, "detail": str(e)}

    # 3. speaker
    try:
        await robot_post("/api/volume/test-sound")
        results["speaker"] = {"pass": True, "detail": "test sound played (did you hear it?)"}
    except Exception as e:
        results["speaker"] = {"pass": False, "detail": str(e)}

    # 4. microphone: read per-mic energy straight from the XMOS audio board
    # while the speaker plays — bypasses AEC/noise-suppression entirely.
    try:
        await robot_post("/api/volume/test-sound")
        peak = [0.0, 0.0, 0.0, 0.0]
        for _ in range(6):
            e = await robot_get("/api/audio/config/parameter/AEC_SPENERGY_VALUES")
            peak = [max(p, abs(v)) for p, v in zip(peak, e.get("values", [0] * 4))]
            await asyncio.sleep(0.4)
        alive = sum(1 for p in peak if p > 0)
        results["microphone"] = {
            "pass": alive > 0,
            "detail": (
                f"{alive}/4 mics show signal energy (peaks={['%.3g' % p for p in peak]})"
                if alive else
                "all 4 mics report ZERO energy at the XMOS chip even during speaker "
                "playback — check the mic array cable/connector to the audio board"
            ),
        }
    except Exception as e:
        results["microphone"] = {"pass": False, "detail": str(e)}

    # 5. camera: specs from daemon + live WebRTC producer advertised
    try:
        specs = await robot_get("/api/camera/specs")
        ok, wdetail = await webrtc_producer_check()
        results["camera"] = {
            "pass": ok,
            "detail": f"detected '{specs['name']}', "
                      f"{len(specs['available_resolutions'])} modes, {wdetail} — "
                      "see live view in the Robot Eyes panel",
        }
    except Exception as e:
        results["camera"] = {"pass": False, "detail": str(e)}

    # 6. wifi
    try:
        w = await robot_get("/wifi/status")
        results["wifi"] = {"pass": True, "detail": json.dumps(w)[:120]}
    except Exception as e:
        results["wifi"] = {"pass": False, "detail": str(e)}

    return results


# -------------------------------------------------------------- voice agent
# Must come before the catch-all static mount below, or "/" swallows it.

from voice_agent import router as voice_router  # noqa: E402

app.include_router(voice_router)


# ------------------------------------------------------------------- static

app.mount("/", StaticFiles(directory="static", html=True), name="static")
