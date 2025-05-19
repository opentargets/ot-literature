"""Entity dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import Window

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset
from src.literature.method.nlp_pipeline import NLPPipeline

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
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
    def normalise_entities(df: DataFrame) -> DataFrame:
        """Normalise entities using NLP pipeline.

        The output column selected is determined by the NLP pipeline type specified.

        Args:
            df (DataFrame): DataFrame containing entity labels to be normalised.

        Returns:
            DataFrame: DataFrame with additional column containing normalised entity labels.
        """
        normalised_entities = NLPPipeline.apply_pipeline(df)

        return (
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
        )

    @staticmethod
    def get_relevant_entity_ids(df: DataFrame) -> DataFrame:
        """Get relevant entity ids for each entity label.

        Args:
            df (DataFrame): DataFrame containing all entity ids for each entity label.

        Returns:
            DataFrame: DataFrame containing only the relevant entity ids for each entity label.

        """
        w = Window.partitionBy("entityType", "entityLabelNormalised").orderBy(f.col("entityScore").desc())

        return (
            df
            .withColumn("entityRank", f.dense_rank().over(w))
            .filter(f.col("entityRank") == 1)
            .groupBy(f.col("entityType"), f.col("entityLabelNormalised"))
            .agg(f.collect_set(f.col("entityId")).alias("entityIds"))
        )
