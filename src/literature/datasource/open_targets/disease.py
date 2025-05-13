"""Open Targets disease index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session
from src.literature.dataset.entity import Entity

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsDisease:
    """Class to process Open Targets disease index."""

    @staticmethod
    def _extract_entities(disease_index: DataFrame) -> DataFrame:
        """Process the Open Targets disease index to extract disease entities.
        
        Args:
            disease_index (DataFrame): DataFrame with the disease index.

        Returns:
            Entity: Extracted disease entities.
        """
        return (
            disease_index
            # select relevant fields
            .select(
                f.col("id").alias("entityId"), 
                f.col("name"), 
                f.col("synonyms.*")
            )
            # annotate entity with score and nlpPipelineType
            .withColumn(
                "name", 
                Entity.annotate_entity(
                    f.array(f.col("name")), 
                    f.lit(1.0), 
                    f.lit("term")
                )
            )
            .withColumn(
                "exactSynonyms", 
                Entity.annotate_entity(
                    f.col("hasExactSynonym"), 
                    f.lit(0.999), 
                    f.lit("term")
                )
            )
            .withColumn(
                "narrowSynonyms", 
                Entity.annotate_entity(
                    f.col("hasNarrowSynonym"), 
                    f.lit(0.998), 
                    f.lit("term")
                )
            )
            .withColumn(
                "broadSynonyms", 
                Entity.annotate_entity(
                    f.col("hasBroadSynonym"), 
                    f.lit(0.997), 
                    f.lit("term")
                )
            )
            .withColumn(
                "relatedSynonyms", 
                Entity.annotate_entity(
                    f.col("hasRelatedSynonym"), 
                    f.lit(0.996), 
                    f.lit("term")
                )
            )
            # flatten and explode array of structs
            .withColumn(
                "entity",
                f.explode(
                    f.flatten(
                        f.array(
                            f.col("name"),
                            f.col("broadSynonyms"),
                            f.col("exactSynonyms"),
                            f.col("narrowSynonyms"),
                            f.col("relatedSynonyms")
                        )
                    )
                )
            )
            # select relevant fields and specify entity type
            .select(
                f.col("entityId"),
                f.col("entity.entityLabel").alias("entityLabel"),
                f.col("entity.entityScore").alias("entityScore"),
                f.col("entity.nlpPipelineType").alias("nlpPipelineType"),
                f.lit("DS").alias("entityType")
            )
            # cleanup
            .filter((f.col("entityLabel").isNotNull()) & (f.length(f.col("entityLabel")) > 0))
            .distinct()
        )

    @classmethod
    def from_index(
        cls: type[OpenTargetsDisease], 
        session: Session, 
        disease_index_path: str
    ) -> DataFrame:
        """Get disease entities from the Open Targets disease index.

        Args:
            session (Session): Session object.
            disease_index_path (str): Path to the disease index.

        Returns:
            Entity: Disease entities.
        """
        disease_index = session.spark.read.parquet(disease_index_path)
        return cls._extract_entities(disease_index)
