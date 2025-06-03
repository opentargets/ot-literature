"""OnToma class for ontology mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import Window

from src.literature.method.ontoma.index_parsers import (
    extract_disease_entities,
    extract_target_entities,
    extract_drug_entities
)
from src.literature.method.ontoma.nlp_pipeline import NLPPipeline

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass
class OnToma:
    """Class to initialise an entity lookup table for mapping entities."""

    disease_index: DataFrame | None = None
    target_index: DataFrame | None = None
    drug_index: DataFrame | None = None
    _entity_lut: DataFrame | None = field(init=False, default=None)

    def __post_init__(self: OnToma) -> None:
        """Post init.

        Initialises an entity lookup table for mapping entities using the index(es) provided.

        Raises:
            ValueError: If no index is provided.
        """
        # validate the input
        if (
            self.disease_index is None 
            and self.target_index is None 
            and self.drug_index is None
        ):
            raise ValueError("At least one index must be provided.")
    
        # extract entities to generate entity lookup tables using index-specific functions
        entity_luts = self._extract_index_entities()

        # concatenate entity lookup tables for downstream processing
        self._entity_lut = self._concatenate_entity_luts(entity_luts)

        # normalise the entity lookup table using an NLP pipeline
        self._entity_lut = self._normalise_entities(self._entity_lut)

        # post-processing to get relevant entity ids for each entity label
        self._entity_lut = self._get_relevant_entity_ids(self._entity_lut)

    @property
    def df(self: OnToma) -> DataFrame:
        """Entity lookup table initialised in the post init.

        Returns:
            DataFrame: Entity lookup table initialised in the post init.
        """
        return self._entity_lut
        
    def _extract_index_entities(self: OnToma) -> list[DataFrame]:
        """Extract entities to generate entity lookup tables using functions specific for each index.

        Returns:
            list[DataFrame]: List of entity lookup tables containing extracted entities.
        """
        # specify function to be used for each index
        index_function_dict = {
            "disease_index": (self.disease_index, extract_disease_entities),
            "target_index": (self.target_index, extract_target_entities),
            "drug_index": (self.drug_index, extract_drug_entities)
        }

        return [
            function(index) 
            for name, (index, function) in index_function_dict.items() 
            if index is not None
        ]
    
    @staticmethod
    def _concatenate_entity_luts(lut_list: list[DataFrame]) -> DataFrame:
        """Concatenate entity lookup tables.

        Args:
            lut_list (list[DataFrame]): List of entity lookup tables to be concatenated.

        Returns:
            DataFrame: Concatenated entity lookup table.
        """
        if len(lut_list) == 1:
            return lut_list[0]
        
        return reduce(lambda lut1, lut2: lut1.unionByName(lut2), lut_list)

    @staticmethod
    def _normalise_entities(df: DataFrame) -> DataFrame:
        """Normalise entities using NLP pipeline.

        The output column selected is determined by the NLP pipeline type specified.

        Args:
            df (DataFrame): DataFrame containing entity labels to be normalised.

        Returns:
            DataFrame: DataFrame with additional column containing normalised entity labels.
        """
        normalised_entities = NLPPipeline.apply_pipeline(df, "entityLabel")

        return (
            normalised_entities
            .withColumn(
                "entityLabelNormalised",
                f.when(
                    f.col("nlpPipelineTrack") == "term",
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
                    f.col("nlpPipelineTrack") == "symbol",
                    f.array_join(
                        f.filter(
                            f.col("finished_symbol"), 
                            lambda c: c.isNotNull() & (c != "")
                        ),
                        ""
                    )
                )
            )
            .drop("finished_term", "finished_symbol", "nlpPipelineTrack", "entityLabel")
            .filter(f.col("entityLabelNormalised").isNotNull() & (f.length("entityLabelNormalised") > 0))
        )
    
    @staticmethod
    def _get_relevant_entity_ids(df: DataFrame) -> DataFrame:
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
    
    @staticmethod
    def _extract_input_entities(
            df: DataFrame,
            label_col_name: str
        ) -> DataFrame:
        """Extract entities from the provided dataframe.

        Entities are set up for normalisation via both the term and symbol tracks of the nlp pipeline.

        Args:
            df (DataFrame): DataFrame containing entity labels to be extracted.
            label_col_name (str): Name of the column containing the entity labels.
        
        Returns:
            DataFrame: DataFrame with additional columns containing entity label and NLP pipeline track.
        """
        return (
            df
            .withColumns(
                {
                    # convert greek alphabet to english alphabet
                    # https://www.rapidtables.com/math/symbols/greek_alphabet.html
                    "entityLabel": f.translate(
                        f.col(label_col_name), 
                        "αβγδεζηικλμνξπτυω", 
                        "abgdezhiklmnxptuo"
                    ),
                    # all input entities will be normalised using both the term and symbol tracks of the nlp pipeline
                    "nlpPipelineTrack": f.explode(f.array(f.lit("term"), f.lit("symbol")))
                }
            )
        )

    def map_entities(
            self: OnToma, 
            df: DataFrame, 
            label_col_name: str, 
            type_col_name: str, 
            result_col_name: str
     ) -> DataFrame:
        """Map entities using the entity lookup table.

        Logic:
        1. Extract entities from input dataframe.
        2. Normalise entities using both tracks of the NLP pipeline.
        3. Join with entity lookup table.
        4. Aggregate results from both tracks.

        Args:
            df (DataFrame): DataFrame containing entity labels to be mapped.
            label_col_name (str): Name of the column containing the entity labels.
            type_col_name (str): Name of the column containing the type of the entity label.
            result_col_name (str): Name of the column for the result.

        Returns:
            DataFrame: DataFrame with additional column containing a list of relevant entity ids for each entity label.
        """
        # extract entities from input dataframe
        extracted_entities = self._extract_input_entities(df, label_col_name)

        # normalise entities and join with entity lookup table
        mapped_entities = (
            self._normalise_entities(extracted_entities)
            .join(
                (
                    self.df
                    .select(
                        f.col("entityLabelNormalised"),
                        f.col("entityType").alias(type_col_name),
                        f.col("entityIds")
                    )
                ),
                on=["entityLabelNormalised", type_col_name],
                how="left"
            )
        )

        # aggregate results from both tracks
        return (
            mapped_entities
            .groupBy(df.columns)
            .agg(f.array_distinct(f.flatten(f.collect_set(f.col("entityIds")))).alias(result_col_name))
            # replace empty list with null
            .withColumn(
                result_col_name,
                f.when(f.size(result_col_name) == 0, None)
                .otherwise(f.col(result_col_name))
            )
        )
