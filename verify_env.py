#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 0 / M0 环境验收脚本
========================

验收标准（来自《进度追踪表》M0）:
    1. torch.cuda.is_available() == True
    2. GPU 利用率 > 50%
    3. torchvision 可用

用法:
    conda activate torch
    python verify_env.py

退出码: 0 = 全部通过   1 = 有未通过项
"""

import platform
import shutil
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- 输出工具

OK, BAD, WARN = "✅ 通过", "❌ 失败", "⚠️  警告"
_results = {}


def section(title):
    print(f"\n{'─' * 62}\n  {title}\n{'─' * 62}")


def record(key, passed, detail=""):
    _results[key] = passed
    tag = OK if passed else BAD
    print(f"  [{tag}] {key}" + (f"\n           {detail}" if detail else ""))


def info(label, value):
    print(f"    {label:<26} {value}")


# ---------------------------------------------------------------- GPU 探测

def nvidia_smi_utilization():
    """返回当前 GPU 利用率百分比；取不到则返回 None。"""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        first = out.stdout.strip().splitlines()
        return float(first[0]) if first else None
    except Exception:
        return None


def sample_utilization(stop_event, samples):
    """后台线程：持续采样 GPU 利用率。"""
    while not stop_event.is_set():
        u = nvidia_smi_utilization()
        if u is not None:
            samples.append(u)
        stop_event.wait(0.25)


# ---------------------------------------------------------------- 主流程

def main():
    print("=" * 62)
    print("  阶段 0 / M0 环境验收")
    print("=" * 62)

    # ---------- 1. 基础环境 ----------
    section("1. 基础环境")
    info("操作系统", f"{platform.system()} {platform.release()}")
    info("Python 版本", sys.version.split()[0])
    info("Python 路径", sys.executable)
    info("conda 环境", __import__("os").environ.get("CONDA_DEFAULT_ENV", "(未激活)"))

    # ---------- 2. PyTorch ----------
    section("2. PyTorch / 依赖")
    try:
        import torch
    except ImportError as e:
        record("import torch", False, str(e))
        return finish()

    info("torch 版本", torch.__version__)
    info("CUDA 编译版本", torch.version.cuda or "(CPU-only 版本!)")
    info("cuDNN 版本", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "(不可用)")

    try:
        import torchvision
        info("torchvision 版本", torchvision.__version__)
        record("torchvision 可用", True)
    except Exception as e:
        record("torchvision 可用", False, f"{type(e).__name__}: {e}")

    try:
        import numpy
        info("numpy 版本", numpy.__version__)
    except Exception:
        pass

    if torch.version.cuda is None:
        record("torch 是 CUDA 版本", False, "装成了 CPU-only 版本，需重装 CUDA 版")
    else:
        record("torch 是 CUDA 版本", True)

    # ---------- 3. CUDA 可用性（M0 核心标准①）----------
    section("3. CUDA 可用性  ← M0 核心标准 ①")
    available = torch.cuda.is_available()
    record("torch.cuda.is_available()", available,
           "" if available else
           "GPU 不可用。排查顺序: nvidia-smi → 驱动 → /dev/nvidia* 设备节点")

    if not available:
        info("torch 看到的 GPU 数", torch.cuda.device_count())
        return finish()

    count = torch.cuda.device_count()
    info("GPU 数量", count)
    info("当前设备", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    info("计算能力 (SM)", f"{cap[0]}.{cap[1]}")
    props = torch.cuda.get_device_properties(0)
    info("显存", f"{props.total_memory / 1024**3:.2f} GB")
    info("是否支持 bf16", torch.cuda.is_bf16_supported())

    # ---------- 4. 真实计算 + 利用率采样（M0 核心标准②）----------
    section("4. 真实 GPU 计算 + 利用率采样  ← M0 核心标准 ②")

    # 4a. 正确性: 一次矩阵乘法并校验结果
    try:
        torch.cuda.synchronize()
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(c).all().item())
        record("GPU 矩阵乘法能算出有限值", finite)
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        record("GPU 矩阵乘法", False, f"{type(e).__name__}: {e}")
        return finish()

    # 4b. 持续负载 + 采样利用率
    duration = 6.0
    print(f"    正在跑 {duration:.0f} 秒 GPU 负载并采样利用率 ...")
    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_utilization,
                               args=(stop_event, samples), daemon=True)
    sampler.start()

    t0 = time.time()
    try:
        x = torch.randn(4096, 4096, device="cuda")
        y = torch.randn(4096, 4096, device="cuda")
        iters = 0
        while time.time() - t0 < duration:
            for _ in range(20):
                x = x @ y
                iters += 1
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        stop_event.set()
        sampler.join(timeout=2)
        record("GPU 持续负载", False, f"{type(e).__name__}: {e}")
        return finish()
    finally:
        stop_event.set()
        sampler.join(timeout=2)

    tflops = (iters * 2 * 4096**3) / elapsed / 1e12
    info("完成矩阵乘法次数", f"{iters}")
    info("耗时", f"{elapsed:.2f} s")
    info("等效算力", f"{tflops:.2f} TFLOPS")
    record("GPU 持续负载跑完", iters > 0)

    if not samples:
        record("GPU 利用率 > 50%", False,
               "采样不到利用率：nvidia-smi 不可用或无法通信")
    else:
        avg = sum(samples) / len(samples)
        peak = max(samples)
        info("采样次数", len(samples))
        info("平均利用率", f"{avg:.1f} %")
        info("峰值利用率", f"{peak:.1f} %")
        record("GPU 利用率 > 50%", avg > 50,
               f"平均 {avg:.1f}%（阈值 50%）")

    return finish()


def finish():
    section("验收结果")
    total = len(_results)
    passed = sum(1 for v in _results.values() if v)
    for k, v in _results.items():
        print(f"  {'✅' if v else '❌'} {k}")
    print(f"\n  合计: {passed}/{total} 通过")

    if passed == total:
        print("\n  🎉 M0 验收通过！可以在《进度追踪表》里把 M0 标记为「已完成」\n")
        return 0
    print("\n  ⚠️  还有未通过项，先解决再更新 M0 状态\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
