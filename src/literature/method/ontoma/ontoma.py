"""OnToma class for ontology mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING

from src.literature.dataset.entity import Entity
from src.literature.datasource.open_targets.disease import OpenTargetsDisease
from src.literature.datasource.open_targets.target import OpenTargetsTarget
from src.literature.datasource.open_targets.drug import OpenTargetsDrug

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass
class OnToma:
    """Class to initialise an entity lookup table for mapping entities."""

    disease_index: DataFrame | None = None
    target_index: DataFrame | None = None
    drug_index: DataFrame | None = None
    _entity_lut: DataFrame | None = field(init=False, default=None)

    def __post_init__(self: OnToma) -> None:
        """Post init.

        Initialises an entity lookup table for mapping entities using the index(es) provided.

        Raises:
            ValueError: If no index is provided.
        """
        # validate the input
        if (
            self.disease_index is None 
            and self.target_index is None 
            and self.drug_index is None
        ):
            raise ValueError("At least one index must be provided.")
    
        # extract entities to generate entity lookup tables using index-specific methods
        entity_luts = self._extract_entities()

        # concatenate entity lookup tables for downstream processing
        self._entity_lut = self._concatenate_entity_luts(entity_luts)

        # normalise the entity lookup table using an NLP pipeline
        self._entity_lut = Entity.normalise_entities(self._entity_lut)

        # post-processing to get relevant entity ids for each entity label
        self._entity_lut = Entity.get_relevant_entity_ids(self._entity_lut)

    @property
    def df(self: OnToma) -> DataFrame:
        """Entity lookup table initialised in the post init.

        Returns:
            DataFrame: Entity lookup table initialised in the post init.
        """
        return self._entity_lut
        
    def _extract_entities(self: OnToma) -> list[DataFrame]:
        """Extract entities to generate entity lookup tables using methods specific for each index.

        Returns:
            list[DataFrame]: List of entity lookup tables containing extracted entities.
        """
        # specify method to be used for each index
        index_method_dict = {
            "disease_index": (self.disease_index, OpenTargetsDisease._extract_entities),
            "target_index": (self.target_index, OpenTargetsTarget._extract_entities),
            "drug_index": (self.drug_index, OpenTargetsDrug._extract_entities)
        }

        return [
            method(index) 
            for name, (index, method) in index_method_dict.items() 
            if index is not None
        ]
    
    @staticmethod
    def _concatenate_entity_luts(lut_list: list[DataFrame]) -> DataFrame:
        """Concatenate entity lookup tables.

        Args:
            lut_list (list[DataFrame]): List of entity lookup tables to be concatenated.

        Returns:
            DataFrame: Concatenated entity lookup table.
        """
        if len(lut_list) == 1:
            return lut_list[0]
        
        return reduce(lambda lut1, lut2: lut1.unionByName(lut2), lut_list)
