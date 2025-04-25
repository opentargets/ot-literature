"""Class to reuse spark connection."""

from __future__ import annotations

from pyspark.sql import SparkSession


class Session:
    """This class provides a Spark session."""

    def __init__(
            self: Session,
            spark_uri: str = "local[*]",
            app_name: str = "ot-literature",
    ) -> None:
        """Initialises Spark session.
        
            Args:
                spark_uri (str): Spark URI. Defaults to "local[*]".
                app_name (str): Spark application name. Defaults to "ot-literature".
        """
        self.spark = (
            SparkSession.Builder()
            .master(spark_uri)
            .appName(app_name)
            .getOrCreate()
        )
