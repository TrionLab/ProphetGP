import numpy as np

from prophet_gp.features.gauche_adapter import GaucheFeaturizerRegistry
from prophet_gp.features.topo_physchem import (
    FEATURE_NAMES,
    encode_molecule,
    encode_smiles_matrix,
    topo_physchem_featuriser,
)


def test_feature_names_stable() -> None:
    feat = encode_molecule("CCO")
    assert list(feat.keys()) == FEATURE_NAMES


def test_encode_smiles_matrix_shape() -> None:
    matrix = encode_smiles_matrix(["CCO", "c1ccccc1"])
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()


def test_registry_includes_topo_physchem() -> None:
    registry = GaucheFeaturizerRegistry()
    assert "topo_physchem" in registry.available()
    out = registry.featurize(["CCO"], name="topo_physchem")
    assert out.shape == (1, len(FEATURE_NAMES))


def test_topo_physchem_featuriser_matches_matrix() -> None:
    smiles = ["CCO", "CC"]
    assert np.allclose(topo_physchem_featuriser(smiles), encode_smiles_matrix(smiles))
