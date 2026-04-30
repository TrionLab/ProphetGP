from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def _ecfp_featuriser(smiles_list: Iterable[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES for ECFP featurisation: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        rows.append(np.array(fp, dtype=np.float32))
    return np.vstack(rows)


def _rdkit_desc_featuriser(smiles_list: Iterable[str]) -> np.ndarray:
    desc_names = [name for name, _ in Descriptors._descList]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES for RDKit descriptor featurisation: {smi}")
        vals = [float(getattr(Descriptors, name)(mol)) for name in desc_names]
        rows.append(np.array(vals, dtype=np.float32))
    return np.vstack(rows)


class GaucheFeaturizerRegistry:
    """gauche featuriser adapter.

    Note:
    - gauche 버전별 API 차이를 흡수하기 위해 기본 featuriser도 함께 제공한다.
    - 사용 가능한 gauche featuriser를 동적 조회하고, 실패 시 fallback registry를 사용한다.
    """

    def __init__(self):
        self._fallback: Dict[str, Callable[[Iterable[str]], np.ndarray]] = {
            "ecfp_fingerprints": _ecfp_featuriser,
            "rdkit_descriptors": _rdkit_desc_featuriser,
        }
        self._gauche_dynamic: Dict[str, Callable[[Iterable[str]], np.ndarray]] = self._load_gauche_featurisers()

    def available(self) -> List[str]:
        names = set(self._fallback.keys()) | set(self._gauche_dynamic.keys())
        return sorted(names)

    def featurize(self, smiles_list: Iterable[str], name: str) -> np.ndarray:
        if name == "auto":
            name = "ecfp_fingerprints"
        if name in self._gauche_dynamic:
            return self._gauche_dynamic[name](smiles_list)
        if name in self._fallback:
            return self._fallback[name](smiles_list)
        raise ValueError(f"Unknown featuriser: {name}. Available={self.available()}")

    def _load_gauche_featurisers(self) -> Dict[str, Callable[[Iterable[str]], np.ndarray]]:
        registry: Dict[str, Callable[[Iterable[str]], np.ndarray]] = {}
        try:
            import gauche  # noqa: F401
            from gauche import representations as reps
        except Exception:
            return registry

        for attr in dir(reps):
            if attr.startswith("_"):
                continue
            candidate = getattr(reps, attr)
            if not callable(candidate):
                continue

            def _build_wrapper(fn: Callable) -> Callable[[Iterable[str]], np.ndarray]:
                def _wrapped(smiles_list: Iterable[str]) -> np.ndarray:
                    result = fn(list(smiles_list))
                    return np.asarray(result, dtype=np.float32)

                return _wrapped

            registry[attr.lower()] = _build_wrapper(candidate)
        return registry
