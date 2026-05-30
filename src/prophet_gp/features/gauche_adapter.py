from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, Iterable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

from prophet_gp.features.topo_physchem import topo_physchem_featuriser


def _ecfp_featuriser(smiles_list: Iterable[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES for ECFP featurisation: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        rows.append(np.array(fp, dtype=np.float32))
    return np.vstack(rows)


def _morgan_fp_featuriser(
    smiles_list: Iterable[str], radius: int = 2, n_bits: int = 1024
) -> np.ndarray:
    """Morgan (ECFP-style) binary fingerprint with chirality flags.

    gauche 패키지에 동명 함수가 없을 때 사용하는 내장 구현이다.
    RDKit `GetMorganFingerprintAsBitVect(..., useChirality=True)`와 동일한 설정이다.
    """
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES for Morgan fingerprint featurisation: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
            useChirality=True,
        )
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
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
            "morgan_fp": _morgan_fp_featuriser,
            "rdkit_descriptors": _rdkit_desc_featuriser,
            "topo_physchem": topo_physchem_featuriser,
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
            import gauche
            from gauche import representations as reps
        except Exception:
            return registry

        def _build_wrapper(fn: Callable) -> Callable[[Iterable[str]], np.ndarray]:
            def _wrapped(smiles_list: Iterable[str]) -> np.ndarray:
                result = fn(list(smiles_list))
                return self._to_float_matrix(result)

            return _wrapped

        # 1) representations 루트에 직접 노출된 함수
        for attr in dir(reps):
            if attr.startswith("_"):
                continue
            candidate = getattr(reps, attr)
            if inspect.isfunction(candidate):
                registry[attr.lower()] = _build_wrapper(candidate)

        # 2) representations 하위 모듈(fingerprints/strings/graphs 등) 함수까지 순회
        for mod_info in pkgutil.walk_packages(gauche.__path__, prefix="gauche."):
            if not mod_info.name.startswith("gauche.representations."):
                continue
            try:
                module = importlib.import_module(mod_info.name)
            except Exception:
                continue

            for attr_name, candidate in inspect.getmembers(module, inspect.isfunction):
                if attr_name.startswith("_"):
                    continue
                # 외부에서 import된 함수는 제외하고, 해당 모듈에서 정의된 함수만 등록
                if getattr(candidate, "__module__", "") != module.__name__:
                    continue
                key = attr_name.lower()
                if key in registry:
                    continue
                registry[key] = _build_wrapper(candidate)
        return registry

    def _to_float_matrix(self, result) -> np.ndarray:
        # 기본 경로: 숫자형으로 바로 변환 가능한 경우
        try:
            return np.asarray(result, dtype=np.float32)
        except Exception:
            pass

        # gauche molecular_graphs는 networkx.Graph 리스트를 반환하므로,
        # GP 입력으로 사용 가능한 고정 길이 통계 벡터로 변환한다.
        if isinstance(result, list) and result and hasattr(result[0], "number_of_nodes"):
            rows = []
            for graph in result:
                n_nodes = float(graph.number_of_nodes())
                n_edges = float(graph.number_of_edges())
                avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
                density = (
                    (2.0 * n_edges / (n_nodes * (n_nodes - 1.0))) if n_nodes > 1 else 0.0
                )
                atomic_nums = []
                for _, attrs in graph.nodes(data=True):
                    atomic_nums.append(float(attrs.get("atomic_num", 0.0)))
                atomic_mean = float(np.mean(atomic_nums)) if atomic_nums else 0.0
                atomic_std = float(np.std(atomic_nums)) if atomic_nums else 0.0
                rows.append([n_nodes, n_edges, avg_degree, density, atomic_mean, atomic_std])
            return np.asarray(rows, dtype=np.float32)

        raise ValueError(
            "Featuriser output could not be converted to numeric matrix. "
            "Consider selecting a vector featuriser such as ecfp_fingerprints."
        )
