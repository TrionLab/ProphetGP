"""Topological + physicochemical RDKit descriptors for positional isomer sensitivity."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdPartialCharges

DEFAULT_MAX_LAG = 5


def safe_float(x) -> float:
    if x is None:
        return np.nan
    try:
        value = float(x)
        if np.isfinite(value):
            return value
        return np.nan
    except (TypeError, ValueError):
        return np.nan


def graph_distance_stats(mol: Chem.Mol) -> Dict[str, float]:
    dist_mat = Chem.GetDistanceMatrix(mol)
    upper = dist_mat[np.triu_indices_from(dist_mat, k=1)]

    if len(upper) == 0:
        return {
            "topo_dist_mean": np.nan,
            "topo_dist_max": np.nan,
            "topo_dist_std": np.nan,
            "wiener_index": np.nan,
        }

    return {
        "topo_dist_mean": safe_float(np.mean(upper)),
        "topo_dist_max": safe_float(np.max(upper)),
        "topo_dist_std": safe_float(np.std(upper)),
        "wiener_index": safe_float(np.sum(upper)),
    }


def charge_autocorrelation(mol: Chem.Mol, max_lag: int = DEFAULT_MAX_LAG) -> Dict[str, float]:
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = np.array(
            [safe_float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()],
            dtype=np.float64,
        )
    except Exception:
        charges = np.zeros(mol.GetNumAtoms(), dtype=np.float64)

    charges = np.nan_to_num(charges, nan=0.0)
    dist_mat = Chem.GetDistanceMatrix(mol)
    features: Dict[str, float] = {}

    for d in range(1, max_lag + 1):
        vals = []
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                if int(dist_mat[i, j]) == d:
                    vals.append(charges[i] * charges[j])
        features[f"charge_autocorr_d{d}"] = safe_float(np.mean(vals)) if vals else 0.0

    return features


def atom_property_autocorrelation(mol: Chem.Mol, max_lag: int = DEFAULT_MAX_LAG) -> Dict[str, float]:
    dist_mat = Chem.GetDistanceMatrix(mol)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.float64)
    atomic_masses = np.array([atom.GetMass() for atom in mol.GetAtoms()], dtype=np.float64)
    features: Dict[str, float] = {}

    for prop_name, prop_values in {"Z": atomic_numbers, "mass": atomic_masses}.items():
        for d in range(1, max_lag + 1):
            vals = []
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    if int(dist_mat[i, j]) == d:
                        vals.append(prop_values[i] * prop_values[j])
            features[f"{prop_name}_autocorr_d{d}"] = (
                safe_float(np.mean(vals)) if vals else 0.0
            )

    return features


def basic_physchem_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "mol_wt": safe_float(Descriptors.MolWt(mol)),
        "exact_mol_wt": safe_float(Descriptors.ExactMolWt(mol)),
        "logp": safe_float(Crippen.MolLogP(mol)),
        "molar_refractivity": safe_float(Crippen.MolMR(mol)),
        "tpsa": safe_float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": safe_float(Lipinski.NumHDonors(mol)),
        "hba": safe_float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": safe_float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": safe_float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": safe_float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atom_count": safe_float(mol.GetNumHeavyAtoms()),
        "fraction_csp3": safe_float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def topological_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "balaban_j": safe_float(GraphDescriptors.BalabanJ(mol)),
        "bertz_ct": safe_float(GraphDescriptors.BertzCT(mol)),
        "hall_kier_alpha": safe_float(Descriptors.HallKierAlpha(mol)),
        "kappa1": safe_float(Descriptors.Kappa1(mol)),
        "kappa2": safe_float(Descriptors.Kappa2(mol)),
        "kappa3": safe_float(Descriptors.Kappa3(mol)),
        "chi0": safe_float(Descriptors.Chi0(mol)),
        "chi1": safe_float(Descriptors.Chi1(mol)),
        "chi0n": safe_float(Descriptors.Chi0n(mol)),
        "chi1n": safe_float(Descriptors.Chi1n(mol)),
        "chi2n": safe_float(Descriptors.Chi2n(mol)),
        "chi3n": safe_float(Descriptors.Chi3n(mol)),
        "chi4n": safe_float(Descriptors.Chi4n(mol)),
        "chi0v": safe_float(Descriptors.Chi0v(mol)),
        "chi1v": safe_float(Descriptors.Chi1v(mol)),
        "chi2v": safe_float(Descriptors.Chi2v(mol)),
        "chi3v": safe_float(Descriptors.Chi3v(mol)),
        "chi4v": safe_float(Descriptors.Chi4v(mol)),
    }


def feature_names(max_lag: int = DEFAULT_MAX_LAG) -> List[str]:
    """Stable column order for featuriser output."""
    names: List[str] = []
    names.extend(basic_physchem_descriptors(Chem.MolFromSmiles("C")).keys())
    names.extend(topological_descriptors(Chem.MolFromSmiles("C")).keys())
    names.extend(graph_distance_stats(Chem.MolFromSmiles("C")).keys())
    names.extend(atom_property_autocorrelation(Chem.MolFromSmiles("C"), max_lag=max_lag).keys())
    names.extend(charge_autocorrelation(Chem.MolFromSmiles("C"), max_lag=max_lag).keys())
    return names


# Module load: fixed order for DEFAULT_MAX_LAG.
FEATURE_NAMES: List[str] = feature_names(DEFAULT_MAX_LAG)


def encode_molecule(smiles: str, max_lag: int = DEFAULT_MAX_LAG) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    features: Dict[str, float] = {}
    features.update(basic_physchem_descriptors(mol))
    features.update(topological_descriptors(mol))
    features.update(graph_distance_stats(mol))
    features.update(atom_property_autocorrelation(mol, max_lag=max_lag))
    features.update(charge_autocorrelation(mol, max_lag=max_lag))
    return features


def encode_smiles_matrix(
    smiles_list: Iterable[str],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    nan_fill: float = 0.0,
) -> np.ndarray:
    """Encode SMILES list to a float32 matrix with fixed feature order."""
    if max_lag != DEFAULT_MAX_LAG:
        keys = feature_names(max_lag)
    else:
        keys = FEATURE_NAMES

    rows = []
    for smi in smiles_list:
        feat = encode_molecule(smi, max_lag=max_lag)
        row = np.array([feat[k] for k in keys], dtype=np.float32)
        rows.append(np.nan_to_num(row, nan=nan_fill, posinf=nan_fill, neginf=nan_fill))
    return np.vstack(rows)


def topo_physchem_featuriser(smiles_list: Iterable[str]) -> np.ndarray:
    return encode_smiles_matrix(smiles_list)
