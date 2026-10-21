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


def load_version(version) -> Optional[LoadedModel]:
    if version is None:
        return None
    if version.model_type == reg.TYPE_BASELINE_MEDIAN:
        return LoadedModel(version.version_id, version.model_family, version.model_type, None)
    path = reg.resolve_artifact_path(version.artifact_uri)
    if path is None:
        logger.error("Model %s: artifact '%s' not found - version unusable.", version.version_id, version.artifact_uri)
        return None
    try:
        bundle = load_bundle(path, version.feature_schema_hash)
    except Exception as e:
        logger.error("Model %s: failed to load artifact (%s) - version unusable.", version.version_id, e)
        return None
    return LoadedModel(version.version_id, version.model_family, version.model_type, bundle)


class ModelProvider:
    """Production + shadow models (resolved once per scoring run)."""

    def __init__(self, production: Optional[LoadedModel] = None, shadow: Optional[LoadedModel] = None, resolved: bool = False):
        self.production = production
        self.shadow = shadow
        # resolved=False means "no registry consulted": engine keeps legacy behaviour.
        self.resolved = resolved

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
        return cls(production, shadow, resolved=True)

    def describe(self) -> Dict[str, Any]:
        return {
            "resolved_from_registry": self.resolved,
            "production": self.production.version_id if self.production else None,
            "production_type": self.production.model_type if self.production else None,
            "shadow": self.shadow.version_id if self.shadow else None,
        }
