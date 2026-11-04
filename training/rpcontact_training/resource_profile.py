"""Phase timings and CUDA allocator peaks; independent of optimization settings."""
import csv
import json
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch


class ResourceProfiler:
    def __init__(self, output_path, device):
        self.output = Path(output_path)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.rows = []
        self.configuration = {}
        self.peaks = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        self.started = time.perf_counter()

    def cuda_ready(self):
        return self.device.type == "cuda" and torch.cuda.is_initialized()

    def synchronize(self):
        if self.cuda_ready():
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                pass  # Preserve the original training exception on CUDA failures.

    def capture(self):
        if not self.cuda_ready():
            return {key: None for key in self.peaks}
        result = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }
        for key, value in result.items():
            self.peaks[key] = max(self.peaks[key] or 0, value)
        return result

    @contextmanager
    def phase(self, phase, epoch=None, tag=None):
        self.synchronize()
        self.capture()  # Keep initialization/checkpoint peaks before resetting.
        if self.cuda_ready():
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        status = "completed"
        error = None
        try:
            yield
        except BaseException as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.synchronize()
            row = {"phase": phase, "epoch": epoch, "tag": tag,
                   "wall_seconds": time.perf_counter() - started,
                   "status": status, "error": error, **self.capture()}
            self.rows.append(row)
            with (self.output / "resource_phases.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def finish(self, status, error=None):
        self.synchronize()
        self.capture()
        training = [r["wall_seconds"] for r in self.rows
                    if r["phase"] == "train" and r["status"] == "completed"]
        evaluation = [r["wall_seconds"] for r in self.rows if r["phase"] == "evaluation"]
        summary = {
            "status": status, "error": error, "device": str(self.device),
            "configuration": self.configuration,
            "total_wall_seconds": time.perf_counter() - self.started,
            "training_seconds": sum(training), "evaluation_seconds": sum(evaluation),
            "completed_training_epochs": len(training),
            "mean_training_epoch_seconds": sum(training) / len(training) if training else None,
            **self.peaks,
            "memory_scope": "PyTorch allocator in this process; includes resident tensors; excludes CUDA context/non-PyTorch allocations",
            "timing_scope": "total includes loading, training, evaluation and checkpoint I/O; phase timing synchronizes CUDA",
        }
        if self.cuda_ready():
            summary.update(gpu_name=torch.cuda.get_device_name(self.device),
                           pytorch_version=torch.__version__,
                           cuda_version=torch.version.cuda)
        for key, value in self.peaks.items():
            summary[key.replace("_bytes", "_gib")] = value / 1024**3 if value is not None else None
        (self.output / "resource_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        with (self.output / "resource_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)


def profile_training(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._resource_profiler = ResourceProfiler(self.args.output_path, self.device)
        self._resource_profiler.configuration = {
            key: getattr(self.args, key) for key in
            ("batchsize", "seed", "lr", "patience", "num_epoches", "accumulation_steps",
             "enforce_data", "input_idlist", "data_dir", "feature_type", "model_in_PATH")
            if hasattr(self.args, key)
        }
        status, error = "completed", None
        try:
            return method(self, *args, **kwargs)
        except BaseException as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._resource_profiler.finish(status, error)
    return wrapped


def profile_phase(phase):
    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            profiler = getattr(self, "_resource_profiler", None)
            if profiler is None:
                return method(self, *args, **kwargs)
            tag = getattr(self, "tag", None) if phase == "evaluation" else None
            with profiler.phase(phase, getattr(self, "_resource_epoch", None), tag):
                return method(self, *args, **kwargs)
        return wrapped
    return decorator
