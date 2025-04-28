"""Match dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset
from src.literature.dataset.entity import Entity

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class Match(Dataset):
    """ Match dataset.

    This dataset describes matches extracted from a Publication dataset.
    """

    @classmethod
    def get_schema(cls: type[Match]) -> StructType:
        """Provides the schema for the Match dataset.

        Returns:
            StructType: Schema for the Match dataset.
        """
        return parse_spark_schema("match.json")
    
    def extract_entities(self: Match) -> Entity:
        """Extract entities from matches for normalisation using NLP.
        
        Returns:
            Entity: Entity dataset.
        """
        return Entity(
            _df=(
                self.df
                .select(f.explode("matches").alias("match"))
                .select(
                    f.col("match.label").alias("entityLabelFromSource"), 
                    f.col("match.type").alias("entityType")
                )
                # convert greek alphabet to english alphabet
                # https://www.rapidtables.com/math/symbols/greek_alphabet.html
                .withColumn(
                    "entityLabelTranslated", 
                    f.translate(
                        f.col("entityLabelFromSource"), 
                        "αβγδεζηικλμνξπτυω", 
                        "abgdezhiklmnxptuo"
                    )
                )
                # create array of structs depending on nlpPipelineType
                .withColumn(
                    "entities",
                    f.when(f.col("entityType") == "DS",
                        f.array(
                            f.struct(
                                f.col("entityLabelTranslated").alias("entityLabel"), 
                                f.lit("term").alias("nlpPipelineType")
                            )
                        )
                    )
                    .when(f.col("entityType").isin("GP", "CD"),
                        f.array(
                            f.struct(
                                f.col("entityLabelTranslated").alias("entityLabel"), 
                                f.lit("term").alias("nlpPipelineType")
                            ),
                            f.struct(
                                f.col("entityLabelTranslated").alias("entityLabel"), 
                                f.lit("symbol").alias("nlpPipelineType")
                            )
                        )
                    )
                )
                .withColumn("entity", f.explode("entities"))
                .select(
                    f.col("entityLabelFromSource"), 
                    f.col("entityType"), 
                    f.col("entity.entityLabel").alias("entityLabel"),
                    f.col("entity.nlpPipelineType").alias("nlpPipelineType")
                )
            ),
            _schema=Entity.get_schema()
        )
    