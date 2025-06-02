"""Publication dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset
from src.literature.dataset.match import Match

if TYPE_CHECKING:
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
