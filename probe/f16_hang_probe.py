"""Child process for the kornia#5493 probe: one CPU float16 convolution scenario per invocation.

Modes:
  info                          torch / oneDNN configuration
  conv DTYPE DATA               the 12 separable-blur shapes of ScalePyramid, pure torch (DATA: rand | zeros)
  kornia [save]                 ScaleSpaceDetector on the kornia test input; "save" stores every conv2d input
  replay DIR last|all [DTYPE]   re-run saved conv2d inputs (exact tensors), optionally cast to DTYPE
  replay-rand DIR last          same shape as the last saved conv2d input, fresh random values
"""

import pathlib
import sys
import time

import torch
import torch.nn.functional as F

SHAPES = [(h, k) for h in (96, 48, 24) for k in (11, 13, 17, 21)]


def log(msg: str) -> None:
    print(msg, flush=True)


def mode_info() -> None:
    log(f"torch {torch.__version__} threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    log(f"cpu capability {torch.backends.cpu.get_cpu_capability()}")
    log(f"mkldnn available {torch.backends.mkldnn.is_available()} enabled {torch.backends.mkldnn.enabled}")
    for name in ("_is_mkldnn_fp16_supported", "_is_mkldnn_bf16_supported"):
        try:
            log(f"{name} {getattr(torch.ops.mkldnn, name)()}")
        except Exception as exc:  # noqa: BLE001
            log(f"{name} unavailable: {exc}")
    log(torch.__config__.show())


def blur(x: torch.Tensor, k: int, w: torch.Tensor) -> torch.Tensor:
    # Same two calls as kornia's ScalePyramid._blur_fast (kornia/geometry/transform/pyramid.py).
    c = x.shape[1]
    k_h = w.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    tmp = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), k_h, groups=c)
    k_v = w.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
    return F.conv2d(F.pad(tmp, (0, 0, k // 2, k // 2), mode="reflect"), k_v, groups=c)


def mode_conv(dtype_name: str, data: str) -> None:
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(3)
    for h, k in SHAPES:
        x = torch.rand(1, 3, h, h, dtype=dtype) if data == "rand" else torch.zeros(1, 3, h, h, dtype=dtype)
        w = torch.rand(k, dtype=dtype)
        t = time.perf_counter()
        log(f"conv h={h} k={k} start")
        blur(x, k, w)
        log(f"conv h={h} k={k} ok {time.perf_counter() - t:.4f}s")


def mode_kornia(save: bool, out: pathlib.Path) -> None:
    import kornia
    import kornia.geometry.transform.pyramid as pyr
    from kornia.feature import BlobHessian, ScaleSpaceDetector

    log(f"kornia {kornia.__file__}")
    orig = pyr.F.conv2d
    count = [0]
    convs = out / "convs"
    if save:
        convs.mkdir(parents=True, exist_ok=True)

    def traced(x, w, *args, **kwargs):
        count[0] += 1
        i = count[0]
        if save:
            torch.save({"x": x, "w": w, "args": args, "kwargs": kwargs}, convs / f"{i:03d}.pt")
        log(f"conv#{i} x={tuple(x.shape)} stride={x.stride()} w={tuple(w.shape)} kw={kwargs} start")
        y = orig(x, w, *args, **kwargs)
        log(f"conv#{i} ok")
        return y

    pyr.F.conv2d = traced

    class ColorResponse(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.response = BlobHessian()

        def forward(self, x: torch.Tensor, _sigmas: torch.Tensor) -> torch.Tensor:
            return self.response(x.mean(dim=1, keepdim=True))

    # tests/feature/test_scale_space_detector.py::TestScaleSpaceDetector::test_color_input_with_single_channel_response
    torch.manual_seed(3)
    inp = torch.rand(1, 3, 96, 96, dtype=torch.float16)
    det = ScaleSpaceDetector(20, resp_module=ColorResponse()).to("cpu", torch.float16)
    lafs, responses = det(inp)
    log(f"detector ok lafs={tuple(lafs.shape)} responses={tuple(responses.shape)}")


def saved_files(directory: str, which: str) -> list:
    files = sorted(pathlib.Path(directory).glob("*.pt"))
    return files[-1:] if which == "last" else files


def mode_replay(directory: str, which: str, dtype_name: str | None) -> None:
    for f in saved_files(directory, which):
        d = torch.load(f)
        x, w = d["x"], d["w"]
        if dtype_name:
            x, w = x.to(getattr(torch, dtype_name)), w.to(getattr(torch, dtype_name))
        log(f"replay {f.name} x={tuple(x.shape)} {x.dtype} stride={x.stride()} w={tuple(w.shape)} {d['kwargs']} start")
        F.conv2d(x, w, *d["args"], **d["kwargs"])
        log(f"replay {f.name} ok")


def mode_replay_rand(directory: str) -> None:
    (f,) = saved_files(directory, "last")
    d = torch.load(f)
    torch.manual_seed(0)
    x, w = torch.rand_like(d["x"]), torch.rand_like(d["w"])
    log(f"replay-rand {f.name} x={tuple(x.shape)} w={tuple(w.shape)} {d['kwargs']} start")
    F.conv2d(x, w, *d["args"], **d["kwargs"])
    log(f"replay-rand {f.name} ok")


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "info":
        mode_info()
    elif mode == "conv":
        mode_conv(rest[0], rest[1])
    elif mode == "kornia":
        mode_kornia(bool(rest) and rest[0] == "save", pathlib.Path(rest[1] if len(rest) > 1 else "probe-out"))
    elif mode == "replay":
        mode_replay(rest[0], rest[1], rest[2] if len(rest) > 2 else None)
    elif mode == "replay-rand":
        mode_replay_rand(rest[0])
    else:
        raise SystemExit(f"unknown mode {mode}")
