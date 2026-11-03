"""Resolve which model serves predictions, and load it safely.

The engine never hard-codes a model file. It asks a ``ModelProvider`` for the production model
(and optional shadow model) of a family. Every prediction then records ``model_version`` and
``prediction_source`` so accuracy can be attributed to a version.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from refillcare.ml import registry as reg

logger = logging.getLogger("refillcare.ml.provider")

_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    version_id: str
    model_family: str
    model_type: str
    bundle: Optional[Dict[str, Any]]  # None for baselines

    @property
    def is_baseline(self) -> bool:
        return self.model_type == reg.TYPE_BASELINE_MEDIAN


def load_bundle(path: Path, expected_schema_hash: Optional[str] = None) -> Dict[str, Any]:
    """Load (and cache by path + mtime) a joblib bundle; verify its feature schema."""
    import joblib

    key = (str(path.resolve()), path.stat().st_mtime)
    with _CACHE_LOCK:
        bundle = _CACHE.get(key)
    if bundle is None:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
            raise ModelArtifactError(f"{path.name}: not a RefillCare model bundle (needs 'model' and 'feature_cols').")
        with _CACHE_LOCK:
            _CACHE[key] = bundle
    if expected_schema_hash and reg.feature_schema_hash(bundle["feature_cols"]) != expected_schema_hash:
        raise ModelArtifactError(f"{path.name}: feature schema hash does not match the registry entry.")
    return bundle


# version_id -> why it could not be loaded in this process (shown in the UI / health warning)
LOAD_ERRORS: Dict[str, str] = {}


def _runtime() -> str:
    import sys

    try:
        import sklearn

        skl = sklearn.__version__
    except Exception:  # pragma: no cover
        skl = "?"
    return f"Python {sys.version.split()[0]} / scikit-learn {skl} ({sys.executable})"


def load_version(version) -> Optional[LoadedModel]:
    if version is None:
        return None
    if version.model_type == reg.TYPE_BASELINE_MEDIAN:
        return LoadedModel(version.version_id, version.model_family, version.model_type, None)
    from refillcare.ml.artifact_store import library_versions, local_path_for

    try:
        path = local_path_for(version.artifact_uri, getattr(version, "artifact_sha256", None))
    except Exception as e:  # checksum mismatch / S3 unavailable: never load an unverified pickle
        logger.error("Model %s: artifact unavailable or failed verification (%s) - version unusable.", version.version_id, e)
        LOAD_ERRORS[version.version_id] = f"artifact unavailable or checksum mismatch: {e}"
        return None
    if path is None:
        logger.error("Model %s: artifact '%s' not found - version unusable.", version.version_id, version.artifact_uri)
        LOAD_ERRORS[version.version_id] = f"model file '{version.artifact_uri}' not found"
        return None
    trained_on = (getattr(version, "library_versions", None) or {}).get("scikit_learn")
    if trained_on and trained_on != library_versions()["scikit_learn"]:
        logger.warning("Model %s was trained with scikit-learn %s but %s is installed: re-evaluate before trusting it.",
                       version.version_id, trained_on, library_versions()["scikit_learn"])
    try:
        bundle = load_bundle(path, version.feature_schema_hash)
    except Exception as e:
        logger.error("Model %s: failed to load artifact (%s) - version unusable.", version.version_id, e)
        trained = trained_on or "a different version"
        LOAD_ERRORS[version.version_id] = (f"it was trained with scikit-learn {trained}, but this app is running on "
                                           f"{_runtime()}" if trained != library_versions()["scikit_learn"]
                                           else f"the model file could not be read ({str(e)[:160]})")
        return None
    LOAD_ERRORS.pop(version.version_id, None)
    return LoadedModel(version.version_id, version.model_family, version.model_type, bundle)


class ModelProvider:
    """Production + shadow models (resolved once per scoring run)."""

    def __init__(self, production: Optional[LoadedModel] = None, shadow: Optional[LoadedModel] = None, resolved: bool = False):
        self.production = production
        self.shadow = shadow
        # resolved=False means "no registry consulted": engine keeps legacy behaviour.
        self.resolved = resolved
        # Set when the registry's production model could not be loaded and the baseline serves instead
        self.fallback_from: Optional[str] = None

    @classmethod
    def legacy(cls) -> "ModelProvider":
        return cls(resolved=False)

    @classmethod
    def baseline(cls) -> "ModelProvider":
        return cls(LoadedModel(reg.BASELINE_VERSION_ID, reg.FAMILY_PATH_A_INTERVAL, reg.TYPE_BASELINE_MEDIAN, None), resolved=True)

    @classmethod
    def from_registry(cls, db: Session, family: str = reg.FAMILY_PATH_A_INTERVAL) -> "ModelProvider":
        reg.ensure_bootstrap_versions(db)
        prod_v = reg.get_by_status(db, family, reg.STATUS_PRODUCTION)
        production = load_version(prod_v)
        if prod_v is not None and production is None:
            logger.error("Production model %s unusable; falling back to the baseline policy.", prod_v.version_id)
            production = LoadedModel(reg.BASELINE_VERSION_ID, family, reg.TYPE_BASELINE_MEDIAN, None)
        shadow = load_version(reg.get_by_status(db, family, reg.STATUS_SHADOW))
        provider = cls(production, shadow, resolved=True)
        if prod_v is not None and production.version_id != prod_v.version_id:
            provider.fallback_from = prod_v.version_id
        return provider

    def describe(self) -> Dict[str, Any]:
        return {
            "resolved_from_registry": self.resolved,
            "production": self.production.version_id if self.production else None,
            "production_type": self.production.model_type if self.production else None,
            "shadow": self.shadow.version_id if self.shadow else None,
            "fallback_from": self.fallback_from,
            "degraded": bool(self.fallback_from),
            "load_error": LOAD_ERRORS.get(self.fallback_from) if self.fallback_from else None,
            "warning": (f"Production model {self.fallback_from} could not be loaded: "
                        f"{LOAD_ERRORS.get(self.fallback_from, 'unknown reason')}. Predictions use the baseline engine.")
                       if self.fallback_from else None,
        }
