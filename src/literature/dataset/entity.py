"""Entity dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql.types import StructType


@dataclass
class Entity(Dataset):
    """Entity dataset.

    This dataset describes entities extracted from the literature and from various indices.
    """

    @classmethod
    def get_schema(cls: type[Entity]) -> StructType:
        """Provides the schema for the Entity dataset.

        Returns:
            StructType: Schema for the Entity dataset.
        """
        return parse_spark_schema("entity.json")
    
    @staticmethod
    def annotate_entity(c: Column, entity_score: float, nlp_pipeline_type: str) -> Column:
        """Annotate entity with score and the NLP pipeline to be processed with.
        
        Args:
            c (Column): Column containing entity label.
            entity_score (float): Score of the entity.
            nlp_pipeline_type (str): NLP pipeline type to be used.
        
        Returns:
            Column: Column of annotated entities.
        """
        return f.transform(
            # Replace null with empty array
            f.coalesce(c, f.array()),
            lambda x: f.struct(
                x.alias("entityLabel"),
                f.lit(entity_score).alias("entityScore"),
                f.lit(nlp_pipeline_type).alias("nlpPipelineType")
            )
        )
