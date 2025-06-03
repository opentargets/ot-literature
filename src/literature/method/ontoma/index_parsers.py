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
                f.array(f.col("name")), 1.0, "term"
            ).alias("nameTerm"),
            _annotate_entity(
                f.array(f.col("name")), 1.0, "symbol"
            ).alias("nameSymbol"),
            _annotate_entity(
                f.col("synonyms.hasExactSynonym"), 0.999, "term"
            ).alias("exactSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasExactSynonym"), 0.999, "symbol"
            ).alias("exactSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasNarrowSynonym"), 0.998, "term"
            ).alias("narrowSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasNarrowSynonym"), 0.998, "symbol"
            ).alias("narrowSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasBroadSynonym"), 0.997, "term"
            ).alias("broadSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasBroadSynonym"), 0.997, "symbol"
            ).alias("broadSynonymsSymbol"),
            _annotate_entity(
                f.col("synonyms.hasRelatedSynonym"), 0.996, "term"
            ).alias("relatedSynonymsTerm"),
            _annotate_entity(
                f.col("synonyms.hasRelatedSynonym"), 0.996, "symbol"
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
        .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
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
                f.array(f.col("approvedName")), 1.0, "term"
            ).alias("name"),
            _annotate_entity(
                f.array(f.col("approvedSymbol")), 1.0, "symbol"
            ).alias("symbol"),
            _annotate_entity(
                f.col("nameSynonyms.label"), 0.999, "term"
            ).alias("nameSynonyms"),
            _annotate_entity(
                f.col("symbolSynonyms.label"), 0.999, "symbol"
            ).alias("symbolSynonyms"),
            _annotate_entity(
                f.col("proteinIds.id"), 0.999, "symbol"
            ).alias("proteinIds"),
            _annotate_entity(
                f.col("obsoleteNames.label"), 0.998, "term"
            ).alias("obsoleteNames"),
            _annotate_entity(
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
                f.array(f.col("name")), 1.0, "term"
            ).alias("nameTerm"),
            _annotate_entity(
                f.array(f.col("name")), 1.0, "symbol"
            ).alias("nameSymbol"),
            _annotate_entity(
                f.col("tradeNames"), 0.999, "term"
            ).alias("tradeNamesTerm"),
            _annotate_entity(
                f.col("tradeNames"), 0.999, "symbol"
            ).alias("tradeNamesSymbol"),
            _annotate_entity(
                f.col("synonyms"), 0.999, "term"
            ).alias("synonymsTerm"),
            _annotate_entity(
                f.col("synonyms"), 0.999, "symbol"
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
        .filter((f.col("entityLabel").isNotNull()) & (f.length("entityLabel") > 0))
        .distinct()
    )
