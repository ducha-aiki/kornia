"""Driver for the kornia#5493 probe: run each scenario in a fresh process; on a hang, collect native stacks."""

import os
import pathlib
import signal
import subprocess
import sys
import time

OUT = pathlib.Path(os.environ.get("PROBE_OUT", "probe-out")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
PROBE = str(pathlib.Path(__file__).with_name("f16_hang_probe.py"))
PY = {"2.9.1": "/tmp/v291/bin/python", "2.14.0": "/tmp/v214/bin/python"}
CAP = {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}
results: list = []


def sh(cmd: list, path: pathlib.Path, timeout: int) -> None:
    with path.open("w") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            f.write(f"\n[timed out after {timeout}s]\n")


def ticks(pid: int) -> tuple:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], int(fields[11]) + int(fields[12])  # state, utime + stime


def diagnose(pid: int, name: str) -> None:
    state1, t1 = ticks(pid)
    time.sleep(5)
    state2, t2 = ticks(pid)
    hz = os.sysconf("SC_CLK_TCK")
    (OUT / f"{name}.proc.txt").write_text(
        f"state {state1} -> {state2}; cpu {(t2 - t1) / hz:.2f}s over 5s wall (1 thread busy = 5.00)\n"
        + pathlib.Path(f"/proc/{pid}/status").read_text()
    )
    gdb = ["sudo", "gdb", "-p", str(pid), "-batch", "-ex", "set pagination off", "-ex", "info threads"]
    sh(gdb + ["-ex", "thread apply all bt 60"], OUT / f"{name}.gdb1.txt", 180)
    time.sleep(3)
    sh(gdb + ["-ex", "thread apply all bt 25"], OUT / f"{name}.gdb2.txt", 180)
    sh(["sudo", "/tmp/v291/bin/py-spy", "dump", "--native", "--pid", str(pid)], OUT / f"{name}.pyspy.txt", 120)


def run(name: str, cmd: list, env: dict | None = None, timeout: int = 60) -> str:
    full_env = dict(os.environ, **(env or {}))
    start = time.time()
    with (OUT / f"{name}.log").open("w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n# env {env or {}}\n")
        logf.flush()
        p = subprocess.Popen(cmd, env=full_env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
            status = "ok" if rc == 0 else f"exit {rc}"
        except subprocess.TimeoutExpired:
            status = "HANG"
            diagnose(p.pid, name)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    last = [ln for ln in (OUT / f"{name}.log").read_text().splitlines() if ln.strip()][-1:]
    line = f"RESULT {name:<28} {status:<8} {time.time() - start:6.1f}s  last: {last[0][:150] if last else ''}"
    print(line, flush=True)
    results.append((name, status, cmd, env))
    return status


def main() -> None:
    py, py14 = PY["2.9.1"], PY["2.14.0"]
    run("info-2.9.1", [py, PROBE, "info"])
    run("info-2.9.1-cap", [py, PROBE, "info"], CAP)
    run("info-2.14.0", [py14, PROBE, "info"])
    run("conv-f16-rand", [py, PROBE, "conv", "float16", "rand"])
    run("conv-f16-zeros", [py, PROBE, "conv", "float16", "zeros"])
    run("kornia-plain", [py, PROBE, "kornia", "plain", str(OUT)])
    run("kornia-save", [py, PROBE, "kornia", "save", str(OUT)])
    convs = str(OUT / "convs")
    if list(pathlib.Path(convs).glob("*.pt")):
        run("replay-last", [py, PROBE, "replay", convs, "last"])
        run("replay-all", [py, PROBE, "replay", convs, "all"])
    pytest = [py, "-m", "pytest", "tests/feature/test_scale_space_detector.py", "--dtype=float16"]
    pytest += ["-p", "no:cacheprovider", "-q", "-o", "faulthandler_timeout=0"]
    run("pytest-file", pytest, timeout=240)

    order = ["replay-last", "replay-all", "conv-f16-rand", "conv-f16-zeros", "kornia-save", "kornia-plain", "pytest-file"]
    hung = {name: (cmd, env) for name, status, cmd, env in results if status == "HANG"}
    for name in order:
        if name not in hung:
            continue
        cmd, _ = hung[name]
        timeout = 240 if name == "pytest-file" else 60
        run(f"{name}-again", cmd, timeout=timeout)
        run(f"{name}-verbose", cmd, {"ONEDNN_VERBOSE": "all"}, timeout=timeout)
        run(f"{name}-cap", cmd, CAP, timeout=timeout)
        run(f"{name}-omp4", cmd, {"OMP_NUM_THREADS": "4"}, timeout=timeout)
        if name != "pytest-file":
            run(f"{name}-torch2.14", [py14] + cmd[1:], timeout=timeout)
        if name.startswith("replay"):
            run(f"{name}-f32", cmd + ["float32"], timeout=timeout)
        if name == "replay-last":
            run("replay-rand-last", [py, PROBE, "replay-rand", convs, "last"], timeout=timeout)
        break  # controls for the most minimal hanging scenario only

    summary = "\n".join(f"{n}: {s}" for n, s, _, _ in results)
    (OUT / "summary.txt").write_text(summary + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("```\n" + summary + "\n```\n")


if __name__ == "__main__":
    sys.exit(main())
