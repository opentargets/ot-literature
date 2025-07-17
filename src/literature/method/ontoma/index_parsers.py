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


def extract_drug_entities(drug_index: DataFrame) -> DataFrame:
    """Process the Open Targets drug index to extract drug entities.
    
    Args:
        drug_index (DataFrame): DataFrame with the drug index.

    Returns:
        DataFrame: DataFrame with the extracted drug entities.
    """
    return (
        drug_index
         # filter crossReferences for sources that have labels
        .withColumn(
            "crossReferences", 
            f.filter(
                f.col("crossReferences"),
                lambda x: x["source"].isin("DailyMed", "USAN", "EMA")
            )
        )
        # transform array of structs to array of strings and format ids
        .withColumn(
            "crossReferences",
            f.transform(
                f.col("crossReferences"),
                lambda x: f.when(
                    # if it's a DailyMed or USAN id, replace spaces encoded as "%20"
                    x["source"].isin("DailyMed", "USAN"),
                    f.transform(x["ids"], lambda i: f.regexp_replace(i, "%20", " "))
                ).when(
                    # if it's an EMA id, extract the last part
                    x["source"] == "EMA",
                    f.transform(x["ids"], lambda i: f.regexp_extract(i, r'.+/EPAR/(.+)', 1))
                ).otherwise(x["ids"])
            )
        )
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            annotate_entity(
                f.array(f.col("name")), 1.0, "term"
            ).alias("nameTerm"),
            annotate_entity(
                f.array(f.col("name")), 1.0, "symbol"
            ).alias("nameSymbol"),
            annotate_entity(
                f.col("tradeNames"), 0.999, "term"
            ).alias("tradeNamesTerm"),
            annotate_entity(
                f.col("tradeNames"), 0.999, "symbol"
            ).alias("tradeNamesSymbol"),
            annotate_entity(
                f.col("synonyms"), 0.999, "term"
            ).alias("synonymsTerm"),
            annotate_entity(
                f.col("synonyms"), 0.999, "symbol"
            ).alias("synonymsSymbol"),
            annotate_entity(
                f.flatten(f.col("crossReferences")), 0.998, "term"
            ).alias("crossReferencesTerm"),
            annotate_entity(
                f.flatten(f.col("crossReferences")), 0.998, "symbol"
            ).alias("crossReferencesSymbol")
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
                        f.col("synonymsSymbol"),
                        f.col("crossReferencesTerm"),
                        f.col("crossReferencesSymbol")
                    )
                )
            )
        )
        # select relevant fields and specify entity type
        .select(
            f.col("entityId"),
            translate_special_characters(
                f.trim(
                    f.col("entity.entityLabel")
                )
            ).alias("entityLabel"),
            f.col("entity.entityScore").alias("entityScore"),
            f.col("entity.nlpPipelineTrack").alias("nlpPipelineTrack"),
            f.lit("CD").alias("entityType")
        )
        # cleanup
        .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
        .distinct()
    )

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

def as_drug_id_lut(drug_index: DataFrame) -> DataFrame:
    """Generate drug id lookup table from the Open Targets drug index.

    Args:
        drug_index (DataFrame): Open Targets drug index.

    Returns:
        DataFrame: Drug id lookup table.
    """
    return (
        drug_index
        # filter crossReferences for sources that have ids
        .withColumn(
            "crossReferences", 
            f.filter(
                f.col("crossReferences"),
                lambda x: x["source"].isin("chEBI", "drugbank")
            )
        )
        # transform array of structs to array of strings and format ids
        .withColumn(
            "crossReferences",
            f.transform(
                f.col("crossReferences"),
                lambda x: f.when(
                    # if it's a chEBI id, append "CHEBI" as a prefix
                    x["source"] == "chEBI",
                    f.concat(f.lit("CHEBI"), x["ids"][0])
                ).otherwise(x["ids"][0])
            )
        )
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            annotate_entity(
                f.col("crossReferences"), 1.0, "symbol"
            ).alias("crossReferences")
        )
        # explode array of structs
        .withColumn(
            "entity",
            f.explode(
                f.col("crossReferences"),
            )
        )
        # select relevant fields and specify entity type
        .select(
            f.col("entityId"),
            f.col("entity.entityLabel").alias("entityLabel"),
            f.col("entity.entityScore").alias("entityScore"),
            f.col("entity.nlpPipelineTrack").alias("nlpPipelineTrack"),
            f.lit("CD").alias("entityType")
        )
        # cleanup
        .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
        .distinct()
    )
