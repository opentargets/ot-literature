"""Open Targets drug index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session
from src.literature.dataset.entity import Entity

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsDrug:
    """Class to process Open Targets drug index."""

    @staticmethod
    def _extract_entities(drug_index: DataFrame) -> Entity:
        """Process the Open Targets drug index to extract drug entities.
        
        Args:
            drug_index (DataFrame): DataFrame with the drug index.

        Returns:
            Entity: Extracted drug entities.
        """
        return Entity(
            _df=(
                drug_index
                # select relevant fields
                .select(
                    f.col("id").alias("entityId"),
                    f.col("name"),
                    f.col("tradeNames"),
                    f.col("synonyms")
                )
                # annotate entity with score and nlpPipelineType
                .withColumn(
                    "nameTerm", 
                    Entity.annotate_entity(
                        f.array(f.col("name")), 
                        f.lit(1.0), 
                        f.lit("term")
                    )
                )
                .withColumn(
                    "nameSymbol", 
                    Entity.annotate_entity(
                        f.array(f.col("name")), 
                        f.lit(1.0), 
                        f.lit("symbol")
                    )
                )
                .withColumn(
                    "tradeNamesTerm", 
                    Entity.annotate_entity(
                        f.col("tradeNames"), 
                        f.lit(0.999), 
                        f.lit("term")
                    )
                )
                .withColumn(
                    "tradeNamesSymbol", 
                    Entity.annotate_entity(
                        f.col("tradeNames"), 
                        f.lit(0.999), 
                        f.lit("symbol")
                    )
                )
                .withColumn(
                    "synonymsTerm", 
                    Entity.annotate_entity(
                        f.col("synonyms"), 
                        f.lit(0.999), 
                        f.lit("term")
                    )
                )
                .withColumn(
                    "synonymsSymbol", 
                    Entity.annotate_entity(
                        f.col("synonyms"), 
                        f.lit(0.999), 
                        f.lit("symbol")
                    )
                )
                # flatten and explode array of structs
                .withColumn(
                    "entity",
                    f.explode(
                        f.flatten(
                            f.array(
                                f.col("nameTerm"),
                                f.col("nameSymbol"),
                                f.col("tradeNamesTerm"),
                                f.col("tradeNamesSymbol"),
                                f.col("synonymsTerm"),
                                f.col("synonymsSymbol")
                            )
                        )
                    )
                )
                # select relevant fields
                .select(
                    f.col("entityId"),
                    f.col("entity.entityLabel").alias("entityLabel"),
                    f.col("entity.entityScore").alias("entityScore"),
                    f.col("entity.nlpPipelineType").alias("nlpPipelineType")
                )
                .filter((f.col("entityLabel").isNotNull()) & (f.length(f.col("entityLabel")) > 0))
                .distinct()
            ),
            _schema=Entity.get_schema()
        )
    
    @classmethod
    def from_index(
        cls: type[OpenTargetsDrug], 
        session: Session, 
        drug_index_path: str
    ) -> Entity:
        """Get drug entities from the Open Targets drug index.

        Args:
            session (Session): Session object.
            drugindex_path (str): Path to the drug index.

        Returns:
            Entity: Drug entities.
        """
        drug_index = session.spark.read.parquet(drug_index_path)
        return cls._extract_entities(drug_index)
    