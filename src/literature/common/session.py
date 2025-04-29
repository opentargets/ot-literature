"""Class to reuse spark connection."""

from __future__ import annotations

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
        config=SparkConf()

        if for_nlp:
            config = (
                SparkConf()
                .set("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.0")
            )

        self.spark = (
            SparkSession.Builder()
            .config(conf=config)
            .master(spark_uri)
            .appName(app_name)
            .getOrCreate()
        )
