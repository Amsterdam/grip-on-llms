"""Utils related to gathering metadata such as infra or package info"""
import importlib.metadata
import platform
import sys

import psutil
import torch


def get_installed_packages():
    installed_packages = importlib.metadata.distributions()
    installed_packages_list = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in installed_packages
    )
    return installed_packages_list


def get_python_version():
    return sys.version


def get_environment_info():
    environment_info = {
        "package_versions": get_installed_packages(),
        "python_version": get_python_version(),
    }
    return environment_info


def get_memory_human():
    memory_info = psutil.virtual_memory()
    return {
        "total": f"{memory_info.total / (1024 ** 3):.2f} GB",
        "available": f"{memory_info.available / (1024 ** 3):.2f} GB",
        "used": f"{memory_info.used / (1024 ** 3):.2f} GB",
        "percent": f"{memory_info.percent}%",
    }


def get_device_info():
    cuda_available = torch.cuda.is_available()
    device_info = {
        "device": torch.device("cuda" if cuda_available else "cpu"),
        "os": {"system": platform.system(), "release": platform.release()},
    }
    if cuda_available:
        total = torch.cuda.get_device_properties(0).total_memory
        total_human = f"{total / (1024 ** 3):.2f} GB"
        used = torch.cuda.memory_allocated(0)
        used_human = f"{used / (1024 ** 3):.2f} GB"
        utilization_human = f"{used / total * 100:.2f}%"

        device_info["gpu"] = {
            "device_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": total_human,
            "gpu_memory_used": used_human,
            "gpu_memory_utilization": utilization_human,
        }

    device_info["cpu"] = {
        "processor": platform.processor(),
        "cpu_memory": get_memory_human(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_utilization": f"{psutil.cpu_percent(interval=1)}%",
    }

    return device_info
