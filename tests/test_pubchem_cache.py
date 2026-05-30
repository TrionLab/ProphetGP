from pathlib import Path

from prophet_gp.chem.pubchem_cache import PubChemCache


def test_pubchem_cache_roundtrip(tmp_path: Path) -> None:
    cache = PubChemCache(tmp_path / "pubchem")
    assert cache.get("ethanol", "name") is None

    cache.set("ethanol", "name", "CCO")
    assert cache.get("ethanol", "name") == "CCO"
    assert cache.get("ethanol", "cas") is None
    assert cache.get("methanol", "name") is None


def test_pubchem_cache_persists_across_instances(tmp_path: Path) -> None:
    cache_dir = tmp_path / "pubchem"
    PubChemCache(cache_dir).set("aspirin", "name", "CC(=O)Oc1ccccc1C(=O)O")

    assert PubChemCache(cache_dir).get("aspirin", "name") == "CC(=O)Oc1ccccc1C(=O)O"
