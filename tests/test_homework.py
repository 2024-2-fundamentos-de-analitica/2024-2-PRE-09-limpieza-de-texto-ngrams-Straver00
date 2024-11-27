"""Autograding script."""

import os

import pandas as pd  # type: ignore

from homework import clean_data


def test_01():
    """Test homework/clean_data.py"""

    clean_data.main(
        "files/input/input.txt",
        "files/output.txt",
    )

    if not os.path.exists("files/test.csv"):
        raise FileNotFoundError("File 'files/test.csv' not found")

    test = pd.read_csv("files/test.csv", index_col=None)

    assert test.loc[0, "key"] == "acdeghinoqruy"
    assert test.loc[1, "key"] == "acdegilmnoty"
    assert test.loc[3, "key"] == "acdehioqrsu"
    assert test.loc[6, "key"] == "acdehoqruy"
    assert test.loc[12, "key"] == "acdeilmnoty"
    assert test.loc[16, "key"] == "acdgilnoprstu"

    #
    # Retorna error si la carpeta output/ no existe
    if not os.path.exists("files/output.txt"):
        raise FileNotFoundError("File 'files/output.txt' not found")

    #
    # Lee el contenido del archivo output.txt
    dataframe = pd.read_csv("files/output.txt")
    count = dataframe.groupby("text").size()

    assert count.loc["AD-HOC QUERIES"] == 3
    assert count.loc["AGRICULTURAL PRODUCTION"] == 1
    assert count.loc["AIRLINE COMPANIES"] == 1
    assert count.loc["AIRLINES"] == 1
    assert count.loc["ANALYTIC APPLICATIONS"] == 9
    assert count.loc["ANALYTIC MODEL"] == 4
