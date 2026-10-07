"""Class to reuse spark connection."""

from __future__ import annotations

from ontoma import spark_nlp_coordinate
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession


class Session:
    """This class provides a Spark session."""

    def __init__(
            self: Session,
            spark_uri: str = "local[*]",
            app_name: str = "ot-literature",
            for_nlp: bool = False
    ) -> None:
        """Initialises Spark session.
        
            Args:
                spark_uri (str): Spark URI. Defaults to "local[*]".
                app_name (str): Spark application name. Defaults to "ot-literature".
                for_nlp (bool): Whether session is used for sparknlp. Defaults to False.
        """
        config=(
            SparkConf()
            .set("spark.driver.memory", "2g")
            .set("spark.executor.memory", "8g")
        )

        if for_nlp:
            config = (
                SparkConf()
                .set("spark.driver.memory", "2g")
                .set("spark.executor.memory", "8g")
                # Spark NLP artifact matching the installed pyspark (Scala 2.12 for Spark 3, 2.13 for Spark 4)
                .set("spark.jars.packages", spark_nlp_coordinate())
            )

        self.spark = (
            SparkSession.Builder()
            .config(conf=config)
            .master(spark_uri)
            .appName(app_name)
            .getOrCreate()
        )
