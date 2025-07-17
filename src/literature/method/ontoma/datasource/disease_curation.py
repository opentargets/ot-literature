"""Open Targets disease curation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.method.ontoma.utils import (
    annotate_entity,
    translate_special_characters,
    clean_disease_label
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsDiseaseCuration:
    """Class to extract disease entities from the Open Targets disease curation table."""

    @classmethod
    def as_label_lut(
        cls: type[OpenTargetsDiseaseCuration], 
        disease_curation: DataFrame
    ) -> DataFrame:
        """Generate disease label lookup table from the Open Targets disease curation table.

        Args:
            disease_curation (DataFrame): Open Targets disease curation table.
        
        Returns:
            DataFrame: Disease label lookup table.
        """
        return (
            disease_curation
            # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
            .select(
                f.regexp_extract(
                    f.col("SEMANTIC_TAG"), r'^http.+/(\w+_\w+)$', 1
                ).alias("entityId"),
                annotate_entity(
                    f.array(f.col("PROPERTY_VALUE")), 1.0, "term"
                ).alias("curationTerm"),
                annotate_entity(
                    f.array(f.col("PROPERTY_VALUE")), 1.0, "symbol"
                ).alias("curationSymbol")
            )
            # flatten and explode array of structs
            .withColumn(
                "entity",
                f.explode(
                    f.flatten(
                        f.array(
                            f.col("curationTerm"),
                            f.col("curationSymbol")
                        )
                    )
                )
            )
            # select relevant fields and specify entity type
            .select(
                f.col("entityId"),
                clean_disease_label(
                    translate_special_characters(
                        f.trim(
                            f.col("entity.entityLabel")
                        )
                    )
                ).alias("entityLabel"),
                f.col("entity.entityScore").alias("entityScore"),
                f.col("entity.nlpPipelineTrack").alias("nlpPipelineTrack"),
                f.lit("DS").alias("entityType")
            )
            # cleanup
            .filter((f.col("entityId").isNotNull()) & (f.length("entityId") > 0))
            .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
            .distinct()
        )
