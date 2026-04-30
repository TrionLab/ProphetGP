from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import requests
from rdkit import Chem
from rdkit import RDLogger

_RD_LOGGER = RDLogger.logger()
_RD_LOGGER.setLevel(RDLogger.CRITICAL)

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


class MoleculeResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class MoleculeRecord:
    raw: str
    smiles: str


class MoleculeResolver:
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    def to_canonical_smiles(self, value: str) -> str:
        token = value.strip()
        if not token:
            raise MoleculeResolutionError("Empty molecule input is not allowed.")

        as_smiles = self._validate_smiles(token)
        if as_smiles:
            return as_smiles

        query_type = "name"
        if CAS_PATTERN.match(token):
            query_type = "name"

        converted = self._resolve_with_pubchem(token, query_type=query_type)
        canonical = self._validate_smiles(converted)
        if not canonical:
            raise MoleculeResolutionError(f"Unable to canonicalize molecule input: {value}")
        return canonical

    def resolve_many(self, reactants: Iterable[str]) -> List[MoleculeRecord]:
        resolved: List[MoleculeRecord] = []
        for item in reactants:
            smiles = self.to_canonical_smiles(item)
            resolved.append(MoleculeRecord(raw=item, smiles=smiles))
        return resolved

    def _validate_smiles(self, value: str) -> str | None:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _resolve_with_pubchem(self, token: str, query_type: str = "name") -> str:
        url = f"{self.PUBCHEM_BASE}/{query_type}/{token}/property/CanonicalSMILES/TXT"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise MoleculeResolutionError(
                f"Could not resolve molecule '{token}' via PubChem (status={response.status_code})."
            )
        text = response.text.strip()
        if not text:
            raise MoleculeResolutionError(f"PubChem returned empty result for '{token}'.")
        return text.splitlines()[0].strip()
