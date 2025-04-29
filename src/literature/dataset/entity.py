"""Entity dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset
from src.literature.method.nlp_pipeline import NLPPipeline

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

    def normalise_entities(self: Entity) -> Entity:
        """Normalise entities using NLP pipeline.

        The output column selected is determined by the NLP pipeline type specified.
        """
        normalised_entities = NLPPipeline.apply_pipeline(self.df)

        return Entity(
            _df=(
                normalised_entities
                .withColumn(
                    "entityLabelNormalised",
                    f.when(
                        f.col("nlpPipelineType") == "term",
                        f.array_join(
                            f.array_sort(
                                f.filter(
                                    f.array_distinct(f.col("finished_term")),
                                    lambda c: c.isNotNull() & (c != "")
                                )
                            ),
                            ""
                        )
                    ).when(
                        f.col("nlpPipelineType") == "symbol",
                        f.array_join(
                            f.filter(
                                f.col("finished_symbol"), 
                                lambda c: c.isNotNull() & (c != "")
                            ),
                            ""
                        )
                    )
                )
                .drop("finished_term", "finished_symbol")
                .filter(f.col("entityLabelNormalised").isNotNull() & (f.length(f.col("entityLabelNormalised")) > 0))
                .distinct()
            ),
            _schema=Entity.get_schema()
        )
