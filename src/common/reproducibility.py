"""Records what a run was made of: config hash, seed, library versions and the git commit."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata

from src.common.config import CONFIG_PATH, REPORTS_DIR, ROOT, ensure_dirs, get_logger

log = get_logger(__name__)

PACKAGES = ["pandas", "numpy", "pyarrow", "mlflow", "pytest"]


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(cfg):
    ensure_dirs()
    info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "random_seed": cfg["random_seed"],
        "reference_date": cfg["reference_date"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
    }
    (REPORTS_DIR / "run_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("Recorded run metadata (config %s)", info["config_sha256"][:12])
