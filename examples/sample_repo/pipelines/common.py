"""Shared helpers for the sample pipelines."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def dedupe_latest(df: DataFrame, key: str, order_by: str) -> DataFrame:
    """Keep one row per key: the one with the greatest `order_by` value."""
    w = Window.partitionBy(key).orderBy(F.col(order_by).desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def write_table(df: DataFrame, table: str, mode: str = "append") -> None:
    """Write a DataFrame to a Delta table."""
    df.write.format("delta").mode(mode).saveAsTable(table)


def add_audit_columns(df: DataFrame, source: str) -> DataFrame:
    def stamp(frame: DataFrame) -> DataFrame:
        return frame.withColumn("_ingested_at", F.current_timestamp())

    return stamp(df).withColumn("_source", F.lit(source))
