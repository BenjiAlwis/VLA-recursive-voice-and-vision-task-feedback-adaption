# Troubleshooting

**Teleoperate/record suddenly can't find the arm after a replug or reboot**
Port paths (`/dev/tty.usbmodemXXXX` on macOS, `/dev/ttyACM0` on Linux) can
change when a device is unplugged/replugged. Re-run `scripts/01_find_ports.sh`
and update `configs/leader.env` / `configs/follower.env`.

**Camera feed is wrong device or fails to open**
Camera `index_or_path` can also shift after replug (especially with multiple
webcams, or a phone camera app). Re-run `scripts/04_find_cameras.sh` and
update `configs/cameras.env`.

**Calibration seems to silently not apply / arm moves oddly**
Calibration files are keyed by `--robot.id` / `--teleop.id`
(`SO101_FOLLOWER_ID` / `SO101_LEADER_ID`). If these don't match exactly
across `03_calibrate.sh`, `05_teleoperate.sh`, `06_record_dataset.sh`, and
`08_infer_rollout.sh`, the pipeline will load the wrong calibration file (or
none) without an obvious error. Keep the ids fixed in `configs/*.env` and
don't rename them per-session.

**Linux: permission denied opening the serial port**
`sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1` (or add your user to the `dialout`
group and re-login).

**SmolVLA training seems too slow / no GPU available**
Fine-tuning is documented at ~4 hours / 20k steps on an A100. Either reduce
`STEPS` in `.env` (do a smoke test at `STEPS=100` first regardless), or use
HF Jobs for cloud GPU by adding `--job.target=a10g-small` (or similar) to
`scripts/07_train_smolvla.sh`. Don't cut episode count below the ~50-episode
floor to save time — that hurts policy quality more than fewer training
steps does.

**Scary-looking traceback about `libtorchcodec`/`libavutil` on every command**
Benign on macOS without Homebrew ffmpeg installed. LeRobot tries `torchcodec`
for video decoding, fails to load it, and automatically falls back to `pyav`
— this is expected and logged, not an error. Also benign: an
`AttributeError: '_thread.RLock' object has no attribute '_recursion_count'`
from `multiprocess.resource_tracker` printed at process exit — a known
Python 3.12 shutdown quirk, unrelated to this project's code.

**A `lerobot-*` command doesn't exist / has different flags than documented**
The LeRobot CLI changes across releases. Run `lerobot-<command> --help` on
your installed version and cross-check against the current docs at
https://huggingface.co/docs/lerobot — the commands in this repo's scripts
were correct as of when this was written but may drift.
