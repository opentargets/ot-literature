"""Functions to extract entities from Open Targets indices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

__all__ = [
    "extract_disease_entities",
    "extract_target_entities",
    "extract_drug_entities"
]


def _annotate_entity(c: Column, entity_score: float, nlp_pipeline_type: str) -> Column:
    """Annotate entity with score and the NLP pipeline to be processed with.
    
    Args:
        c (Column): Column containing entity label.
        entity_score (float): Score of the entity.
        nlp_pipeline_type (str): NLP pipeline type to be used.
    
    Returns:
        Column: Column of struct of annotated entities.
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

def extract_disease_entities(disease_index: DataFrame) -> DataFrame:
    """Process the Open Targets disease index to extract disease entities.
    
    Args:
        disease_index (DataFrame): DataFrame with the disease index.

    Returns:
        DataFrame: DataFrame with the extracted disease entities.
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
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("term")
            )
        )
        .withColumn(
            "exactSynonyms", 
            _annotate_entity(
                f.col("hasExactSynonym"), 
                f.lit(0.999), 
                f.lit("term")
            )
        )
        .withColumn(
            "narrowSynonyms", 
            _annotate_entity(
                f.col("hasNarrowSynonym"), 
                f.lit(0.998), 
                f.lit("term")
            )
        )
        .withColumn(
            "broadSynonyms", 
            _annotate_entity(
                f.col("hasBroadSynonym"), 
                f.lit(0.997), 
                f.lit("term")
            )
        )
        .withColumn(
            "relatedSynonyms", 
            _annotate_entity(
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

def extract_target_entities(target_index: DataFrame) -> DataFrame:
    """Process the Open Targets target index to extract target entities.
    
    Args:
        target_index (DataFrame): DataFrame with the target index.

    Returns:
        DataFrame: DataFrame with the extracted target entities.
    """
    return (
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
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("term")
            )
        )
        .withColumn(
            "symbol", 
            _annotate_entity(
                f.array(f.col("symbol")), 
                f.lit(1.0), 
                f.lit("symbol")
            )
        )
        .withColumn(
            "nameSynonyms", 
            _annotate_entity(
                f.col("nameSynonyms"), 
                f.lit(0.999), 
                f.lit("term")
            )
        )
        .withColumn(
            "symbolSynonyms", 
            _annotate_entity(
                f.col("symbolSynonyms"), 
                f.lit(0.999), 
                f.lit("symbol")
            )
        )
        .withColumn(
            "proteinIds", 
            _annotate_entity(
                f.col("proteinIds"), 
                f.lit(0.999), 
                f.lit("symbol")
            )
        )
        .withColumn(
            "obsoleteNames", 
            _annotate_entity(
                f.col("obsoleteNames"), 
                f.lit(0.998), 
                f.lit("term")
            )
        )
        .withColumn(
            "obsoleteSymbols", 
            _annotate_entity(
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
        # select relevant fields and specify entity type
        .select(
            f.col("entityId"), 
            f.col("entity.entityLabel").alias("entityLabel"), 
            f.col("entity.entityScore").alias("entityScore"), 
            f.col("entity.nlpPipelineType").alias("nlpPipelineType"),
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
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("term")
            )
        )
        .withColumn(
            "nameSymbol", 
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("symbol")
            )
        )
        .withColumn(
            "tradeNamesTerm", 
            _annotate_entity(
                f.col("tradeNames"), 
                f.lit(0.999), 
                f.lit("term")
            )
        )
        .withColumn(
            "tradeNamesSymbol", 
            _annotate_entity(
                f.col("tradeNames"), 
                f.lit(0.999), 
                f.lit("symbol")
            )
        )
        .withColumn(
            "synonymsTerm", 
            _annotate_entity(
                f.col("synonyms"), 
                f.lit(0.999), 
                f.lit("term")
            )
        )
        .withColumn(
            "synonymsSymbol", 
            _annotate_entity(
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
        # select relevant fields and specify entity type
        .select(
            f.col("entityId"),
            f.col("entity.entityLabel").alias("entityLabel"),
            f.col("entity.entityScore").alias("entityScore"),
            f.col("entity.nlpPipelineType").alias("nlpPipelineType"),
            f.lit("CD").alias("entityType")
        )
        # cleanup
        .filter((f.col("entityLabel").isNotNull()) & (f.length(f.col("entityLabel")) > 0))
        .distinct()
    )
