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


def extract_target_entities(target_index: DataFrame) -> DataFrame:
    """Process the Open Targets target index to extract target entities.
    
    Args:
        target_index (DataFrame): DataFrame with the target index.

    Returns:
        DataFrame: DataFrame with the extracted target entities.
    """
    return (
        target_index
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            annotate_entity(
                f.array(f.col("approvedName")), 1.0, "term"
            ).alias("name"),
            annotate_entity(
                f.array(f.col("approvedSymbol")), 1.0, "symbol"
            ).alias("symbol"),
            annotate_entity(
                f.col("nameSynonyms.label"), 0.999, "term"
            ).alias("nameSynonyms"),
            annotate_entity(
                f.col("symbolSynonyms.label"), 0.999, "symbol"
            ).alias("symbolSynonyms"),
            annotate_entity(
                f.col("proteinIds.id"), 0.999, "symbol"
            ).alias("proteinIds"),
            annotate_entity(
                f.col("obsoleteNames.label"), 0.998, "term"
            ).alias("obsoleteNames"),
            annotate_entity(
                f.col("obsoleteSymbols.label"), 0.998, "symbol"
            ).alias("obsoleteSymbols")
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
            f.lit("GP").alias("entityType")
        )
        # cleanup
        .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
        .distinct()
    )

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

def as_target_id_lut(target_index: DataFrame) -> DataFrame:
    """Generate target id lookup table from the Open Targets target index.

    Args:
        target_index (DataFrame): Open Targets target index.

    Returns:
        DataFrame: Target id lookup table.
    """
    return (
        target_index
        # filter out Xrefs with signalP as a source as only two possible ids (SignalP-TM and SignalP-noTM)
        .withColumn(
            "dbXrefs", 
            f.filter(
                f.col("dbXrefs"),
                lambda x: x["source"] != "signalP"
            )
        )
        # transform array of structs to array of strings and format ids
        .withColumn(
            "dbXrefs",
            f.transform(
                f.col("dbXrefs"),
                lambda x: f.when(
                    # if it's a HGNC id, append "HGNC" as a prefix
                    x["source"] == "HGNC",
                    f.concat(f.lit("HGNC"), x["id"])
                ).otherwise(x["id"])
            )
        )
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            annotate_entity(
                f.col("dbXrefs"), 1.0, "symbol"
            ).alias("dbXrefs"),
            annotate_entity(
                f.col("proteinIds.id"), 1.0, "symbol"
            ).alias("proteinIds")
        )
        # flatten and explode array of structs
        .withColumn(
            "entity",
            f.explode(
                f.flatten(
                    f.array(
                        f.col("dbXrefs"),
                        f.col("proteinIds")
                    )
                )
            )
        )
        # select relevant fields and specify entity type
        .select(
            f.col("entityId"),
            f.col("entity.entityLabel").alias("entityLabel"),
            f.col("entity.entityScore").alias("entityScore"),
            f.col("entity.nlpPipelineTrack").alias("nlpPipelineTrack"),
            f.lit("GP").alias("entityType")
        )
        # cleanup
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
