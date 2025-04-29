"""Publications from Europe PMC Data Source."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pyspark.sql.functions as f

from src.literature.common.schemas import parse_spark_schema
from src.literature.common.session import Session
from src.literature.dataset.publication import Publication


@dataclass
class EPMCPublication:
    """Class to process publications from Europe PMC."""

    # schema specifying desired subset of columns
    defined_schema = parse_spark_schema("publication.json")

    @classmethod
    def from_source(
        cls: type[EPMCPublication],
        session: Session,
        raw_publication_path: str
    ) -> Publication:
        """Read publications from the specified filepath.

        The publication type (abstract or fulltext) is captured from the filepath.

        Args:
            session (Session): Session object.
            raw_publication_path (str): Path to EPMC publications.

        Returns:
            Publication: EPMC publications.
        """
        # extract publication type
        publication_type = re.search(r'/(abstract|fulltext)/', raw_publication_path).group(1)

        return Publication(
            _df=(
                session.spark.read.schema(cls.defined_schema)
                .json(raw_publication_path)
                .withColumn("kind", f.lit(publication_type))
            ),
            _schema=Publication.get_schema()
        )
    