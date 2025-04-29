"""Open Targets target index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session
from src.literature.dataset.entity import Entity

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass
class OpenTargetsTarget:
    """Class to process Open Targets target index."""

    @staticmethod
    def _extract_entities(target_index: DataFrame) -> Entity:
        """Process the Open Targets target index to extract target entities.
        
        Args:
            target_index (DataFrame): DataFrame with the target index.

        Returns:
            Entity: Extracted target entities.
        """
        return Entity(
            _df=(
                target_index
                # select relevant fields
                .select(
                    f.col("id").alias("entityId"),
                    f.col("approvedName").alias("name"),
                    f.col("approvedSymbol").alias("symbol"),
                    f.col("nameSynonyms.label").alias("nameSynonyms"),
                    f.col("symbolSynonyms.label").alias("symbolSynonyms"),
                    f.col("obsoleteNames.label").alias("obsoleteNames"),
                    f.col("obsoleteSymbols.label").alias("obsoleteSymbols"),
                    f.col("proteinIds.id").alias("proteinIds")
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
                    "symbol", 
                    Entity.annotate_entity(
                        f.array(f.col("symbol")), 
                        f.lit(1.0), 
                        f.lit("symbol")
                    )
                )
                .withColumn(
                    "nameSynonyms", 
                    Entity.annotate_entity(
                        f.col("nameSynonyms"), 
                        f.lit(0.999), 
                        f.lit("term")
                    )
                )
                .withColumn(
                    "symbolSynonyms", 
                    Entity.annotate_entity(
                        f.col("symbolSynonyms"), 
                        f.lit(0.999), 
                        f.lit("symbol")
                    )
                )
                .withColumn(
                    "proteinIds", 
                    Entity.annotate_entity(
                        f.col("proteinIds"), 
                        f.lit(0.999), 
                        f.lit("symbol")
                    )
                )
                .withColumn(
                    "obsoleteNames", 
                    Entity.annotate_entity(
                        f.col("obsoleteNames"), 
                        f.lit(0.998), 
                        f.lit("term")
                    )
                )
                .withColumn(
                    "obsoleteSymbols", 
                    Entity.annotate_entity(
                        f.col("obsoleteSymbols"), 
                        f.lit(0.998), 
                        f.lit("symbol")
                    )
                )
                # flatten and explode array of structs
                .withColumn(
                    "entity",
                    f.explode(
                        f.flatten(
                            f.array(
                                f.col("name"),
                                f.col("symbol"),
                                f.col("nameSynonyms"),
                                f.col("symbolSynonyms"),
                                f.col("proteinIds"),
                                f.col("obsoleteNames"),
                                f.col("obsoleteSymbols")
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
                .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
                .distinct()
            ),
            _schema=Entity.get_schema()
        )
    
    @classmethod
    def from_index(
        cls: type[OpenTargetsTarget], 
        session: Session, 
        target_index_path: str
    ) -> Entity:
        """Get target entities from the Open Targets target index.

        Args:
            session (Session): Session object.
            target_index_path (str): Path to the target index.

        Returns:
            Entity: Target entities.
        """
        target_index = session.spark.read.parquet(target_index_path)
        return cls._extract_entities(target_index)
    