"""Open Targets disease index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class OpenTargetsDisease:
    """Class to process Open Targets disease index."""

    @classmethod
    def from_index(
        cls: type[OpenTargetsDisease], 
        session: Session, 
        disease_index_path: str
    ) -> DataFrame:
        """Get disease entities from the Open Targets disease index.

        Args:
            session (Session): Session object.
            disease_index_path (str): Path to the disease index.

        Returns:
            DataFrame: Disease entities.
        """
        disease_index = session.spark.read.parquet(disease_index_path)
        return cls._extract_entities(disease_index)
