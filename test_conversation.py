"""Conversation soak test — drives the real pipeline, no speaker or mic.

Every bug found in live use so far was a *conversational* failure, not a
crash: JSON spoken aloud, a tool result ignored in favour of an invented
story, tools withheld exactly when the user agreed to a lookup. None of them
raise an exception, so nothing catches them except reading the transcript
afterwards.

This replays scripted conversations through the same Brain and the same
gating the agent uses, and asserts on what would have been *said*.

    NVIDIA_API_KEY=... REACHY_AGENT_ENABLED=1 .venv/bin/python test_conversation.py
    ... --quick     skip the slow lookup cases
"""

from __future__ import annotations

import sys
import time

import voice_agent as va


def route(text: str, pending: dict | None):
    """Exactly the gating _respond does, so the test exercises real routing."""
    wants = {"move": bool(va.MOVE_RE.search(text)),
             "web": bool(va.LOOKUP_RE.search(text) or va.AGENT_RE.search(text))}
    direct = wants["move"] or wants["web"]
    inherited = False
    if not direct and va.AFFIRM_RE.search(text) and pending:
        wants, pending, inherited = dict(pending), None, True
    if direct:
        pending = dict(wants)
    allow = wants if (wants["move"] or wants["web"]) else None
    # Must mirror _respond exactly, or the test validates logic that is not
    # the logic that runs.
    return allow, bool(allow), pending, inherited


CHECKS = {
    "no_json": lambda r: not va.looks_like_tool_json(r) and not r.startswith(("{", "[")),
    "not_empty": lambda r: len(r.strip()) > 1,
    "no_tool_names": lambda r: not any(
        n in r for n in ("run_agent", "web_search", "move_head", "turn_body",
                         "play_emotion", "reset_pose")),
    "no_roleplay": lambda r: "*" not in r,
    "speakable": lambda r: not any(c in r for c in "{}[]<>|`"),
}

# (utterance, must_call_tool)
SCRIPTS = [
    ("smalltalk", [
        ("Hey, good morning.", False),
        ("Yes, I did.", False),
        ("Tell me something interesting.", False),
        ("That's nice.", False),
    ]),
    ("movement", [
        ("turn 30 degrees to the right", True),
        ("look up", True),
        ("show me a happy emotion", True),
        ("go back to neutral", True),
    ]),
    ("vague_then_confirm", [
        ("Can you tell me what to do?", False),
        ("I'm feeling a bit stuck today.", False),
        ("What's the latest news in AI?", True),
        ("Absolutely you should.", None),      # inherited; either is fine
    ]),
    ("mixed_context", [
        ("What's the latest news for Pune?", True),
        ("Hey Richie, can you play any songs?", False),
        ("Yeah, can you suggest something to listen to right now?", True),
    ]),
    ("nonsense", [
        ("uh", False),
        ("But I want you to", False),
        ("I want to latest", None),
        ("okay", None),
    ]),
]

SLOW = {"vague_then_confirm", "mixed_context"}


def main() -> int:
    quick = "--quick" in sys.argv
    brain = va.Brain()
    print(f"model: {va.LLM_MODEL}")
    print(f"agent tool: {'on' if va.AGENT_ENABLED else 'off'}   "
          f"web backend: {'yes' if va.web_backend_available() else 'no'}\n")

    failures, turns = [], 0
    for name, script in SCRIPTS:
        if quick and name in SLOW:
            print(f"-- {name} (skipped) --")
            continue
        print(f"-- {name} --")
        history, pending = [], None
        for text, must_tool in script:
            allow, force, pending, inherited = route(text, pending)
            history.append({"role": "user", "content": text})
            called = []
            t0 = time.time()
            try:
                reply = brain.reply(
                    history, "", allow_tools=allow, force_tool=force,
                    on_tool=lambda n, a, r: called.append(n))
            except Exception as e:
                failures.append((name, text, f"EXCEPTION {type(e).__name__}: {e}"))
                print(f"   ! {text[:40]!r} -> EXCEPTION {type(e).__name__}")
                continue
            reply = va.safe_to_say(reply)
            spoken = va.clean_for_speech(reply)
            history.append({"role": "assistant", "content": reply})
            turns += 1

            bad = [k for k, fn in CHECKS.items() if not fn(reply)]
            if must_tool is True and not called:
                bad.append("expected_tool_call")
            if must_tool is False and called:
                bad.append("unexpected_tool_call")
            if not spoken.strip():
                bad.append("nothing_to_speak")
            for b in bad:
                failures.append((name, text, b))
            flag = "FAIL " + ",".join(bad) if bad else "ok"
            tools = f" [{','.join(called)}]" if called else ""
            print(f"   {flag:<34} ({time.time()-t0:4.1f}s){tools} "
                  f"{text[:30]!r} -> {reply[:60]!r}")

    print(f"\n{turns} turns, {len(failures)} failures")
    for name, text, why in failures:
        print(f"  FAIL [{name}] {text[:44]!r}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
