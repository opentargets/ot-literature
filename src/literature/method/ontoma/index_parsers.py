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


def _annotate_entity(c: Column, entity_score: float, nlp_pipeline_track: str) -> Column:
    """Annotate entity with score and the NLP pipeline to be processed with.
    
    Args:
        c (Column): Column containing entity label.
        entity_score (float): Score of the entity.
        nlp_pipeline_track (str): NLP pipeline track to be used.
    
    Returns:
        Column: Column of struct of annotated entities.
    """
    return f.transform(
        # Replace null with empty array
        f.coalesce(c, f.array()),
        lambda x: f.struct(
            x.alias("entityLabel"),
            f.lit(entity_score).alias("entityScore"),
            f.lit(nlp_pipeline_track).alias("nlpPipelineTrack")
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
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("term")
            ).alias("nameTerm"),
            _annotate_entity(
                f.array(f.col("name")), 
                f.lit(1.0), 
                f.lit("symbol")
            ).alias("nameSymbol"),
            _annotate_entity(
                f.col("synonyms.hasExactSynonym"), 
                f.lit(0.999), 
                f.lit("term")
            ).alias("exactSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasExactSynonym"), 
                f.lit(0.999), 
                f.lit("symbol")
            ).alias("exactSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasNarrowSynonym"), 
                f.lit(0.998), 
                f.lit("term")
            ).alias("narrowSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasNarrowSynonym"), 
                f.lit(0.998), 
                f.lit("symbol")
            ).alias("narrowSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasBroadSynonym"), 
                f.lit(0.997), 
                f.lit("term")
            ).alias("broadSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasBroadSynonym"), 
                f.lit(0.997), 
                f.lit("symbol")
            ).alias("broadSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasRelatedSynonym"), 
                f.lit(0.996), 
                f.lit("term")
            ).alias("relatedSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasRelatedSynonym"), 
                f.lit(0.996), 
                f.lit("symbol")
            ).alias("relatedSynonymsSymbol")
        )
        # flatten and explode array of structs
        .withColumn(
            "entity",
            f.explode(
                f.flatten(
                    f.array(
                        f.col("nameTerm"),
                        f.col("nameSymbol"),
                        f.col("exactSynonymsTerm"),
                        f.col("exactSynonymsSymbol"),
                        f.col("narrowSynonymsTerm"),
                        f.col("narrowSynonymsSymbol"),
                        f.col("broadSynonymsTerm"),
                        f.col("broadSynonymsSymbol"),
                        f.col("relatedSynonymsTerm"),
                        f.col("relatedSynonymsSymbol")
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
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            _annotate_entity(
                f.array(f.col("approvedName")), 
                f.lit(1.0), 
                f.lit("term")
            ).alias("name"),
            _annotate_entity(
                f.array(f.col("approvedSymbol")), 
                f.lit(1.0), 
                f.lit("symbol")
            ).alias("symbol"),
            _annotate_entity(
                f.col("nameSynonyms.label"), 
                f.lit(0.999), 
                f.lit("term")
            ).alias("nameSynonyms"),
            _annotate_entity(
                f.col("symbolSynonyms.label"), 
                f.lit(0.999), 
                f.lit("symbol")
            ).alias("symbolSynonyms"),
            _annotate_entity(
                f.col("proteinIds.id"), 
                f.lit(0.999), 
                f.lit("symbol")
            ).alias("proteinIds"),
            _annotate_entity(
                f.col("obsoleteNames.label"), 
                f.lit(0.998), 
                f.lit("term")
            ).alias("obsoleteNames"),
            _annotate_entity(
                f.col("obsoleteSymbols.label"), 
                f.lit(0.998), 
                f.lit("symbol")
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
            f.col("entity.entityLabel").alias("entityLabel"), 
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
        # extract entities from relevant fields and annotate entity with score and nlpPipelineTrack
        .select(
            f.col("id").alias("entityId"),
            _annotate_entity(
                f.array(f.col("name")),
                f.lit(1.0), 
                f.lit("term")
            ).alias("nameTerm"),
            _annotate_entity(
                f.array(f.col("name")),
                f.lit(1.0), 
                f.lit("symbol")
            ).alias("nameSymbol"),
            _annotate_entity(
                f.col("tradeNames"),
                f.lit(0.999), 
                f.lit("term")
            ).alias("tradeNamesTerm"),
            _annotate_entity(
                f.col("tradeNames"),
                f.lit(0.999), 
                f.lit("symbol")
            ).alias("tradeNamesSymbol"),
            _annotate_entity(
                f.col("synonyms"),
                f.lit(0.999), 
                f.lit("term")
            ).alias("synonymsTerm"),
            _annotate_entity(
                f.col("synonyms"),
                f.lit(0.999), 
                f.lit("symbol")
            ).alias("synonymsSymbol")
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
            f.col("entity.nlpPipelineTrack").alias("nlpPipelineTrack"),
            f.lit("CD").alias("entityType")
        )
        # cleanup
        .filter((f.col("entityLabel").isNotNull()) & (f.length(f.col("entityLabel")) > 0))
        .distinct()
    )
