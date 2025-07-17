"""Functions to extract entities from Open Targets indices."""

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

__all__ = [
    "extract_disease_entities",
    "extract_target_entities",
    "extract_drug_entities",
    "extract_disease_curation",
    "as_target_id_lut",
    "as_drug_id_lut"
]


def extract_disease_curation(disease_curation: DataFrame) -> DataFrame:
    """Process a disease curation table to extract disease entities.

    It is expected that the disease curation table contains the SEMANTIC_TAG and PROPERTY_VALUE fields
    as in the format mentioned here: https://github.com/opentargets/curation/blob/master/mappings/disease/README.md

    Args:
        disease_curation (DataFrame): DataFrame with the disease curation.
    
    Returns:
        DataFrame: DataFrame with the extracted disease entities.
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
