"""Utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

if TYPE_CHECKING:
    from pyspark.sql import Column


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
        f.regexp_extract(
            f.element_at(f.split(disease_label, "#"), -1), 
            r'^(?:[A-Z]{1}[0-9]{2}[-.A-Z0-9]* |Chapter [IVX]+ )?(.+)$', 
            1
        )
    )
