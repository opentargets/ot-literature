"""Open Targets target index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.method.ontoma.utils import (
    annotate_entity,
    translate_special_characters
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsTarget:
    """Class to extract target entities from the Open Targets target index."""

    @classmethod
    def as_target_label_lut(
        cls: type[OpenTargetsTarget], 
        target_index: DataFrame
    ) -> DataFrame:
        """Generate target label lookup table from the Open Targets target index.
        
        Args:
            target_index (DataFrame): Open Targets target index.

        Returns:
            DataFrame: Target label lookup table.
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

    @classmethod
    def as_target_id_lut(
        cls: type[OpenTargetsTarget], 
        target_index: DataFrame
    ) -> DataFrame:
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
