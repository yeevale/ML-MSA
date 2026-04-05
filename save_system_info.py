#!/usr/bin/env python3
"""
Save system information before running experiments.
Called automatically from vastai_run.sh.
"""
import json
import os
import sys


def get_system_info() -> dict:
    info = {}

    # Python
    info["Python"] = sys.version.split()[0]

    # PyTorch + CUDA
    try:
        import torch
        info["PyTorch"] = torch.__version__
        info["CUDA_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["VRAM_GB"] = f"{props.total_mem / 1e9:.1f}"
            info["CUDA_version"] = torch.version.cuda
    except Exception:
        pass

    # CPU (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["CPU"] = line.split(":")[1].strip()
                    break
        info["CPU_cores"] = str(os.cpu_count())
        with open("/proc/cpuinfo") as f:
            info["AVX2"] = "yes" if "avx2" in f.read() else "no"
    except Exception:
        info["CPU_cores"] = str(os.cpu_count())

    # RAM (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["RAM_GB"] = f"{kb / 1e6:.1f}"
                    break
    except Exception:
        pass

    # aligner module
    try:
        import aligner
        info["aligner_module"] = "OK"
    except Exception as e:
        info["aligner_module"] = f"ERROR: {e}"

    return info


if __name__ == "__main__":
    info = get_system_info()
    os.makedirs("results", exist_ok=True)
    with open("results/system_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("System info saved to results/system_info.json")
    for k, v in info.items():
        print(f"  {k}: {v}")
