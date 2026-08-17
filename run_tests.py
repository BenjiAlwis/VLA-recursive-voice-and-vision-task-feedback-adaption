"""
Run every module's self-test.  Owner: Benji.

    python run_tests.py           # all of it, ~20s
    python run_tests.py -v        # show the output of failures

Each module in this repo self-tests when run directly. This just runs them
all and reports. There is no pytest and no test/ directory on purpose: at
hackathon pace, a test that lives next to the code it checks and runs with
`python thatfile.py` actually gets run, and a separate suite does not.

Everything here is OFFLINE. No API key, no camera, no arm. If this passes
on a fresh clone, the system works end to end on the mock.
"""
import os
import subprocess
import sys
import time

# Speech and mock dwell-time are stripped: the tests check logic, and 20
# seconds of `say` per run means nobody runs them.
ENV = {**os.environ, "MOCK_REALTIME": "0", "TTS_BACKEND": "print",
       "GLASSES_VIDEO": "0"}

MODULES = [
    ("arm_api", "the contract + mock physics; all 5 failure modes reachable"),
    ("vision", "ArUco ground truth in cm, and occlusion handling"),
    ("schema", "the action menu rejects everything it must"),
    ("critic", "geometry decides pass/fail; a model cannot override it"),
    ("planner", "prompt assembly + the fallback path"),
    ("memory", "failure store, retrieval, the ablation wipe"),
    ("narrate", "speech out + device routing"),
    ("voice", "speech in, and the chain that always terminates"),
    ("glasses", "audio core; video gated off and safe to be absent"),
    ("teach", "demonstration recording, waypoints, the no-write boundary"),
]

# loop.py's self-test is a flag, not a bare run — a bare run would launch
# an actual trial loop.
EXTRA = [(["loop.py", "--self-test"], "loop",
          "harness arithmetic, IPC merge, escalation")]


def run(cmd, label, why, verbose):
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable] + cmd, env=ENV,
                          capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=180)
    dt = time.monotonic() - t0
    ok = proc.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<10} {dt:5.1f}s  {why}")
    if not ok:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        for line in (tail if verbose else tail[-12:]):
            print(f"          {line}")
    return ok


def main():
    verbose = "-v" in sys.argv
    print(f"\nrunning {len(MODULES) + len(EXTRA)} self-tests "
          f"(offline: no key, no camera, no arm)\n")

    results = []
    for name, why in MODULES:
        results.append(run([f"{name}.py"], name, why, verbose))
    for cmd, name, why in EXTRA:
        results.append(run(cmd, name, why, verbose))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    if passed != total:
        print("\nRun with -v for full output of the failures.")
        return 1

    print("""
Everything offline is green. Still UNVERIFIED against hardware:
  - SO101Arm            needs FOLLOWER_PORT + `python arm_api.py --calibrate`
  - teach.py --leader   needs LEADER_PORT
  - vision.py --live    needs the overhead camera and printed markers
  - planner.py --verify needs NEBIUS_API_KEY  <- do this one first
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
