"""Publication ID lookup table from Europe PMC Data Source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass
class PublicationIdLUT:
    """Class to parse publication id lookup table downloaded from http://ftp.ebi.ac.uk/pub/databases/pmc/DOI/PMID_PMCID_DOI.csv.gz"""

    @staticmethod
    def _parser(df: DataFrame) -> DataFrame:
        """Parse the publication id lookup table.

        Args:
            df (DataFrame): DataFrame with the publication id lookup table.

        Returns:
            DataFrame: DataFrame with the parsed publication id lookup table.
        """
        return(
            df
            .select(f.col("PMID").alias("pmid_lut"), f.col("PMCID").alias("pmcid_lut"))
            .filter(f.col("pmid_lut").isNotNull() & f.col("pmcid_lut").isNotNull() & f.col("pmcid_lut").startswith("PMC"))
            .distinct()
        )

    @classmethod
    def from_csv(cls: type[PublicationIdLUT], session: Session, csv_path: str) -> DataFrame:
        """Read publication id lookup table from csv.

        Args:
            session (Session): Session object.
            csv_path (str): Path to the publication id lookup table.

        Returns:
            DataFrame: DataFrame with the publication id lookup table.
        """
        return cls._parser(session.spark.read.csv(csv_path, header=True))
