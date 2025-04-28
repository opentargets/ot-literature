"""Publication dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset
from src.literature.dataset.match import Match

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType


@dataclass
class Publication(Dataset):
    """Publication dataset.
    
    This dataset describes publications which can be in the form of either abstracts or fulltexts.
    """

    @classmethod
    def get_schema(cls: type[Publication]) -> StructType:
        """Provides the schema for the Publication dataset.

        Returns:
            StructType: Schema for the Publication dataset.
        """
        return parse_spark_schema("publication.json")
    
    def process_fulltexts_using_lut(self: Publication, lut: DataFrame) -> Publication:
        """Add pmid information to fulltexts that do not already have pmids using a publication id lookup table.

        Should only be used when Publication dataset contains only fulltexts.
        
        Args:
            lut (DataFrame): DataFrame with the publication id lookup table.
        
        Returns:
            Publication: Publication dataset with pmid information added.
        """
        return Publication(
            _df=(
                self.df.alias("fulltext")
                .join(
                    lut.alias("lut"),
                    on = [
                        (f.col("fulltext.pmcid") == f.col("lut.pmcid_lut"))
                        & (
                            f.col("fulltext.pmid").isNull()
                            | (f.col("fulltext.pmid").isNotNull() 
                            & (f.col("fulltext.pmid") == f.col("lut.pmid_lut")))
                        )
                    ],
                    how="inner"
                )
                .withColumn("pmid", f.coalesce(f.col("pmid"), f.col("pmid_lut")))
                .drop("pmid_lut", "pmcid_lut")
            ),
            _schema=Publication.get_schema()
        )

    def merge_abstracts_with_fulltexts(self: Publication, fulltext: Publication) -> Publication:
        """Merge abstracts with fulltexts, excluding abstracts for which the fulltext version exists.

        Should only be used when Publication dataset contains only abstracts.
        
        Args:
            fulltext (Publication): Fulltext publications.
        
        Returns:
            Publication: Complete set of non-repeating publications consisting of both abstracts and fulltexts.
        """
        # exclude abstracts with fulltexts
        abstracts_without_fulltexts = (
            self.df
            .join(
                fulltext.df,
                on="pmid",
                how="left_anti"
            )
        )

        # combine abstracts with fulltexts
        return Publication(
            _df=(
                abstracts_without_fulltexts
                .unionByName(
                    fulltext.df,
                    allowMissingColumns=True
                )
            ),
            _schema=Publication.get_schema()
        )
    
    def get_most_recent_publications(self: Publication) -> Publication:
        """Deduplicate publications by taking the publication with the most recent timestamp.
        
        Returns:
            Publication: Deduplicated set of publications.
        """
        # add timestamp column
        timestamped = (
            self.df
            .withColumn("int_timestamp", f.unix_timestamp(f.col("timestamp")))
        )

        # get most recent version of each publication
        most_recent_publications = (
            timestamped
            .groupBy(f.col("pmcid"), f.col("pmid"))
            .agg(f.max(f.col("int_timestamp")).alias("max_timestamp"))
        )

        # get other relevant fields for each most recent publication
        return Publication(
            _df=(
                most_recent_publications
                .select(
                    f.col("pmcid").alias("mrpi_pmcid"), 
                    f.col("pmid").alias("mrpi_pmid"), 
                    f.col("max_timestamp")
                )
                .join(
                    timestamped,
                    on=[
                        f.col("mrpi_pmcid").eqNullSafe(timestamped.pmcid) &
                        f.col("mrpi_pmid").eqNullSafe(timestamped.pmid) &
                        f.col("max_timestamp").eqNullSafe(timestamped.int_timestamp)
                    ],
                    how="left"
                )
                .drop("mrpi_pmcid", "mrpi_pmid", "max_timestamp")
            ),
            _schema=Publication.get_schema()
        )

    def extract_matches(self: Publication) -> Match:
        """Extract matches information from publications.
        
        Returns:
            Match: Match dataset.
        """
        return Match(
            _df=(
                self.df
                .withColumn("sentence", f.explode("sentences"))
                .select("*", "sentence.*")
                .drop("sentences", "sentence")
                .withColumn("section", f.lower("section"))
            ),
            _schema=Match.get_schema()
        )
