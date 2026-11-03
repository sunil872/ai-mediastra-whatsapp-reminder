"""Model files (.joblib): per-store, checksummed, optionally in S3, with a small retention window.

- **Where:** local ``<store>/processed/models`` always (fast load / cache); additionally S3
  (``REFILLCARE_MODEL_BUCKET``, prefix ``REFILLCARE_MODEL_PREFIX``, key ``<prefix>/<store>/<file>``)
  so a container restart or a new instance gets the model back.
- **Integrity:** SHA-256 of the file is stored in the registry and verified before ``joblib.load``
  (a pickle is code: never load a file whose hash does not match). Library versions are recorded so a
  model trained on another scikit-learn version is flagged.
- **Cost:** per store only production, shadow and the newest ``REFILLCARE_MODEL_KEEP`` (default 3)
  candidates keep their files; older ones are deleted locally and in S3 (registry rows stay).
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("refillcare.ml.artifacts")


class ArtifactIntegrityError(RuntimeError):
    pass


def library_versions() -> Dict[str, str]:
    import numpy
    import pandas
    import sklearn

    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def models_dir() -> Path:
    """Local model folder of the current store (the default store keeps the registry's MODELS_DIR)."""
    from refillcare.ml import registry as reg
    from refillcare.store import StoreConfig
    from refillcare.tenancy import current_store_id, default_store_id

    d = reg.MODELS_DIR if current_store_id() == default_store_id() else StoreConfig.from_env().processed_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bucket() -> Optional[str]:
    return os.getenv("REFILLCARE_MODEL_BUCKET", "").strip() or None


def _s3():
    import boto3  # optional dependency, only needed with REFILLCARE_MODEL_BUCKET

    return boto3.client("s3")


def _s3_key(filename: str) -> str:
    from refillcare.tenancy import current_store_id

    prefix = os.getenv("REFILLCARE_MODEL_PREFIX", "refillcare/models").strip("/")
    return f"{prefix}/{current_store_id()}/{filename}"


def save_bundle(bundle: Dict[str, Any], filename: str) -> Tuple[str, str]:
    """Write the bundle, return (artifact_uri, sha256). URI is ``s3://bucket/key`` when uploaded."""
    import joblib

    local = models_dir() / filename
    joblib.dump(bundle, local, compress=3)
    digest = sha256_file(local)
    bucket = _bucket()
    if bucket:
        key = _s3_key(filename)
        _s3().upload_file(str(local), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": digest}})
        return f"s3://{bucket}/{key}", digest
    return filename, digest


def local_path_for(uri: Optional[str], sha256: Optional[str] = None) -> Optional[Path]:
    """Local file for an artifact URI (downloads from S3 on a cache miss); verifies the checksum."""
    from refillcare.ml import registry as reg

    if not uri:
        return None
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        local = models_dir() / Path(key).name
        if not local.is_file() or (sha256 and sha256_file(local) != sha256):
            tmp = local.with_suffix(".download")
            _s3().download_file(bucket, key, str(tmp))
            os.replace(tmp, local)
        path: Optional[Path] = local
    else:
        path = reg.resolve_artifact_path(uri, models_dir())
        if path is None:
            path = reg.resolve_artifact_path(uri)
    if path is not None and sha256 and sha256_file(path) != sha256:
        raise ArtifactIntegrityError(f"{path.name}: checksum mismatch - refusing to load (file changed or corrupted).")
    return path


def delete_artifact(uri: Optional[str]) -> None:
    if not uri:
        return
    try:
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            _s3().delete_object(Bucket=bucket, Key=key)
            local = models_dir() / Path(key).name
        else:
            local = models_dir() / uri
        if local.is_file():
            local.unlink()
    except Exception:
        logger.exception("Could not delete artifact %s", uri)


def prune_artifacts(db, keep: Optional[int] = None) -> int:
    """Delete files of old candidate / retired versions (production + shadow + newest N kept)."""
    from database.models import MLModelVersionModel
    from refillcare.ml import registry as reg

    keep = keep if keep is not None else int(os.getenv("REFILLCARE_MODEL_KEEP", "3"))
    rows = (db.query(MLModelVersionModel)
            .filter(MLModelVersionModel.model_type == reg.TYPE_HIST_GB, MLModelVersionModel.artifact_uri.isnot(None))
            .order_by(MLModelVersionModel.created_at.desc()).all())
    kept, removed = 0, 0
    for v in rows:
        if v.status in (reg.STATUS_PRODUCTION, reg.STATUS_SHADOW):
            continue
        if kept < keep:
            kept += 1
            continue
        delete_artifact(v.artifact_uri)
        v.artifact_uri = None
        v.notes = ((v.notes or "") + " [artifact pruned by retention]").strip()
        removed += 1
    if removed:
        db.commit()
    return removed
