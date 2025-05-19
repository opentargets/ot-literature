"""Entity dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class Entity(Dataset):
    """Entity dataset.

    This dataset describes entities extracted from the literature and from various indices.
    """

    @classmethod
    def get_schema(cls: type[Entity]) -> StructType:
        """Provides the schema for the Entity dataset.

        Returns:
            StructType: Schema for the Entity dataset.
        """
        return parse_spark_schema("entity.json")
