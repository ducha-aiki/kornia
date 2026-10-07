"""Round 2 of the kornia#5493 probe: environment, minimal repro, shape sweep, pure-oneDNN versions, CI cap."""

import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(os.environ.get("PROBE_OUT", "probe-out2")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
PY291, PY214 = "/tmp/v291/bin/python", "/tmp/v214/bin/python"
CAP = {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}
DNNL = [p for p in os.environ.get("DNNL_PREFIXES", "").split() if p]
lines: list = []


def native_top(pid: int, path: pathlib.Path) -> None:
    cmd = ["sudo", "gdb", "-p", str(pid), "-batch", "-ex", "set pagination off", "-ex", "bt 14"]
    with path.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=120, check=False)


def run(name: str, cmd: list, env: dict | None = None, timeout: int = 10, stack: bool = False) -> str:
    start = time.time()
    log = OUT / f"{name}.log"
    with log.open("w") as f:
        f.write(f"$ {' '.join(cmd)}\n# env {env or {}}\n")
        f.flush()
        p = subprocess.Popen(cmd, env=dict(os.environ, **(env or {})), stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
            status = "ok" if rc == 0 else f"exit{rc}"
        except subprocess.TimeoutExpired:
            status = "HANG"
            if stack:
                native_top(p.pid, OUT / f"{name}.gdb.txt")
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    tail = [ln for ln in log.read_text().splitlines() if ln.strip() and not ln.startswith(("$ ", "# env"))][-1:]
    line = f"{name:<44} {status:<6} {time.time() - start:5.1f}s  {tail[0][:140] if tail else ''}"
    print(line, flush=True)
    lines.append(line)
    return status


def main() -> None:
    # 1. Environment for the PyTorch issue template.
    for tag, py in (("2.9.1", PY291), ("2.14.0", PY214)):
        run(f"collect_env-{tag}", [py, str(HERE / "collect_env.py")], timeout=180)

    # 2. The minimal repro, as it will appear in the report, plus the two workarounds.
    repro = str(HERE / "repro.py")
    run("repro-2.9.1", [PY291, repro], timeout=30, stack=True)
    run("repro-2.14.0", [PY214, repro], timeout=30, stack=True)
    run("repro-2.9.1-cap", [PY291, repro], CAP, timeout=30)
    run("repro-2.14.0-cap", [PY214, repro], CAP, timeout=30)

    # 3. Shape sweep on torch 2.9.1 (one fresh process per case).
    one = str(HERE / "sweep_one.py")
    ks = range(3, 33, 2)

    def case(dt, c, h, ow, k, orient="h", mkldnn="on", py=PY291):
        return run(f"sweep-{dt}-c{c}-h{h}-ow{ow}-k{k}-{orient}-mkldnn{mkldnn}",
                   [py, one, dt, str(c), str(h), str(ow), str(k), orient, mkldnn])

    for ow in (24, 48, 64, 96, 128):
        for k in ks:
            case("float16", 3, 96, ow, k)
    for c in (1, 2, 4, 8, 16, 32, 64):
        case("float16", c, 96, 96, 17)
    for h in (1, 8, 24, 48):
        case("float16", 3, h, 96, 17)
    for k in ks:
        case("float16", 3, 96, 96, k, orient="v")
    for k in ks:
        case("bfloat16", 3, 96, 96, k)
    case("float16", 3, 96, 96, 17, mkldnn="off")
    case("float16", 3, 96, 96, 17, py=PY214)

    # 4. Pure oneDNN: the same problem without PyTorch, across oneDNN versions.
    for prefix in DNNL:
        tag = pathlib.Path(prefix).name
        exe = f"{prefix}/dw_f16_repro"
        if not pathlib.Path(exe).exists():
            lines.append(f"onednn-{tag}: no build, skipped")
            continue
        for variant in (["infer", "plain", "f16"], ["train", "plain", "f16"], ["infer", "any", "f16"],
                        ["infer", "plain", "bf16"], ["infer", "plain", "f32"]):
            run(f"onednn-{tag}-{'-'.join(variant)}", [exe, *variant], timeout=20, stack=variant[2] == "f16")
        run(f"onednn-{tag}-infer-plain-f16-cap", [exe, "infer", "plain", "f16"], CAP, timeout=20)
        for k in ks:
            run(f"onednn-{tag}-kw{k}", [exe, "infer", "plain", "f16", str(k), "96"], timeout=10)

    # 5. The CI workaround end to end: the kornia test file that hangs in CI, under the cap.
    pytest = [PY291, "-m", "pytest", "tests/feature/test_scale_space_detector.py", "--dtype=float16",
              "-p", "no:cacheprovider", "-q", "-o", "faulthandler_timeout=0"]
    run("pytest-file-cap", pytest, CAP, timeout=300)

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("```\n" + "\n".join(lines) + "\n```\n")


if __name__ == "__main__":
    sys.exit(main())
