"""Utility functions."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


def translate_special_characters(label: Column) -> Column:
    """Translate greek alphabet and accented latin characters into latin alphabet.

    Conversions are based on the following websites:
    https://www.rapidtables.com/math/symbols/greek_alphabet.html
    https://en.wikipedia.org/wiki/Latin-1_Supplement

    Args:
        label (Column): Column containing the label to be translated.

    Returns:
        Column: Column containing the translated label.
    """
    return (
        f.translate(
            label,
            "αβγδεζηικλμνξπρτυωàèìòùáéíóúâêîôûäëïöüÀÈÌÒÙÁÉÍÓÚÂÊÎÔÛÄËÏÖÜãåõøÃÅÕØçñýÇÑÝ",
            "abgdezhiklmnxprtuoaeiouaeiouaeiouaeiouAEIOUAEIOUAEIOUAEIOUaaooAAOOcnyCNY"
        )
    )

def clean_disease_label(disease_label: Column) -> Column:
    """Clean disease label by removing prefixes.

    Args:
        disease_label (Column): Column containing the disease label with prefixes.

    Returns:
        Column: Column containing the disease label with prefixes removed.
    """
    return (
        f.when(
            disease_label.contains("#"),
            f.regexp_extract(
                f.element_at(f.split(disease_label, "#"), -1), 
                r'^(?:[A-Z]{1}[0-9]{2}[-.A-Z0-9]* |Chapter [IVX]+ )?(.+)$', 
                1
            )
        ).otherwise(disease_label)
    )

def filter_disease_crossrefs(disease_df: DataFrame) -> DataFrame:
    """Filter out disease crossrefs with irrelevant prefixes.

    Args:
        disease_df (DataFrame): DataFrame containing disease crossrefs.

    Returns:
        DataFrame: DataFrame containing only the relevant disease crossrefs.
    """
    prefix_list = ["PMID", "DOI:", "ORCID", "PERSON", "ISBN", "WIKIPEDIA", "HTTP", "QUANT", "UM-BBD_PATHWAYID"]

    filter_condition = reduce(
        lambda cond1, cond2: cond1 | cond2,
        [f.col("entityLabel").contains(prefix) for prefix in prefix_list],
        f.lit(False)
    )
    
    return disease_df.filter(~filter_condition)

def format_disease_identifier(disease_identifier: Column) -> Column:
    """Format disease identifier to have consistent formatting.

    Args:
        disease_identifier (Column): Column containing the disease identifier.

    Returns:
        Column: Column containing the formatted disease identifier.
    """
    # ensure consistent formatting across disease identifiers
    disease_identifier = (
        f.when(
            f.length(f.regexp_extract(disease_identifier, r'^.+:(.+_.+)$', 1)) > 1,
            f.regexp_extract(disease_identifier, r'^.+:(.+_.+)$', 1)
        ).otherwise(disease_identifier)
    )
    disease_identifier = f.regexp_replace(disease_identifier, "_", ":")
    
    # ensure Orphanet identifiers are consistent
    return f.regexp_replace(disease_identifier, r'ORDO:|ORPHA:', "ORPHANET:")
