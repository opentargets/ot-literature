"""Open Targets drug index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsDrug:
    """Class to process Open Targets drug index."""
    
    @classmethod
    def from_index(
        cls: type[OpenTargetsDrug], 
        session: Session, 
        drug_index_path: str
    ) -> DataFrame:
        """Get drug entities from the Open Targets drug index.

        Args:
            session (Session): Session object.
            drugindex_path (str): Path to the drug index.

        Returns:
            DataFrame: Drug entities.
        """
        drug_index = session.spark.read.parquet(drug_index_path)
        return cls._extract_entities(drug_index)
    