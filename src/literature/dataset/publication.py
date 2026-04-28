"""Publication dataset."""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from literature.common.schemas import parse_spark_schema
from literature.dataset.dataset import Dataset
from literature.dataset.match import Match

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
        logger.info('extract matches')
        return Match(
            _df=(
                self.df
                .withColumn("section", f.lower("section"))
                # pubDate derivatives
                .withColumn("date", f.to_date("pubDate"))
                .withColumn("year", f.year("date"))
                .withColumn("month", f.month("date"))
                .withColumn("day", f.dayofmonth("date"))
                # explode and expand sentences
                .withColumn("sentence", f.explode("sentences"))
                .select("*", "sentence.*")
                .drop("sentences", "sentence")
                # explode and expand matches
                .withColumn("match", f.explode("matches"))
                .select("*", "match.*")
                .drop("matches", "match")
            ),
            _schema=Match.get_schema()
        )
