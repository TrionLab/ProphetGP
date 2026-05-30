from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_pubchem_cache_dir() -> Path:
    return project_root() / "cache" / "pubchem"


class PubChemCache:
    """File-backed cache for PubChem name/CAS -> canonical SMILES lookups."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = (cache_dir or default_pubchem_cache_dir()).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, token: str, query_type: str) -> Path:
        key = f"{query_type}:{token}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, token: str, query_type: str) -> str | None:
        path = self._entry_path(token, query_type)
        if not path.is_file():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("token") != token or data.get("query_type") != query_type:
            return None
        smiles = data.get("canonical_smiles")
        return smiles if isinstance(smiles, str) and smiles else None

    def set(self, token: str, query_type: str, canonical_smiles: str) -> None:
        path = self._entry_path(token, query_type)
        payload = {
            "token": token,
            "query_type": query_type,
            "canonical_smiles": canonical_smiles,
        }
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
