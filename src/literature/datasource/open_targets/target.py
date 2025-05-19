"""Open Targets target index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsTarget:
    """Class to process Open Targets target index."""
    
    @classmethod
    def from_index(
        cls: type[OpenTargetsTarget], 
        session: Session, 
        target_index_path: str
    ) -> DataFrame:
        """Get target entities from the Open Targets target index.

        Args:
            session (Session): Session object.
            target_index_path (str): Path to the target index.

        Returns:
            DataFrame: Target entities.
        """
        target_index = session.spark.read.parquet(target_index_path)
        return cls._extract_entities(target_index)
    