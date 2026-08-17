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
    ("reason", "VLM diagnosis; refuses to run on a passing verdict"),
    ("planner", "prompt assembly + the fallback path"),
    ("memory", "failure store, retrieval, the ablation wipe"),
    ("narrate", "speech out + device routing"),
    ("voice", "speech in, and the chain that always terminates"),
    ("glasses", "audio routing + photo ingest, both halves"),
    ("teach", "demonstration recording, waypoints, the no-write boundary"),
]

# These three need a flag, not a bare run — each would otherwise run
# forever: loop.py starts a real trial loop, glasses_bridge.py serves HTTP,
# supervise.py polls on an interval with no default duration. Note the flag
# spellings differ (--self-test vs --selftest); that is how they were each
# written, and unifying them is not worth a merge conflict today.
EXTRA = [
    (["loop.py", "--self-test"], "loop",
     "harness arithmetic, IPC merge, escalation"),
    (["glasses_bridge.py", "--selftest"], "glasses_bridge",
     "real HTTP POST against a real server, then exits"),
    (["supervise.py", "--selftest"], "supervise",
     "interval glasses-correction loop, no model and no glasses"),
]


TIMEOUT_S = 90


def run(cmd, label, why, verbose):
    """One module's self-test. A hang is reported as a FAIL, not raised.

    The timeout matters: a module that serves forever or polls on an
    interval will hang here, and letting TimeoutExpired escape kills the
    whole run and reports nothing about the modules that did pass.
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.run([sys.executable] + cmd, env=ENV,
                              capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=TIMEOUT_S)
        ok = proc.returncode == 0
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        ok = False
        output = (f"TIMED OUT after {TIMEOUT_S}s. If this module runs a "
                  f"server or an interval loop, it needs a --selftest flag "
                  f"in EXTRA rather than a bare run.")
    dt = time.monotonic() - t0
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<15} {dt:5.1f}s  {why}")
    if not ok:
        tail = output.strip().splitlines()
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
