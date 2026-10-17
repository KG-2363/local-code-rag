"""Orders pipeline for a fictional online shop: bronze -> silver -> gold."""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from pipelines.common import dedupe_latest, write_table

BRONZE = "shop.bronze.orders_raw"
SILVER = "shop.silver.orders"
GOLD = "shop.gold.daily_revenue"


def load_orders_silver(spark: SparkSession) -> DataFrame:
    """Clean raw order events and keep the latest version of each order."""
    raw = spark.table("shop.bronze.orders_raw")
    typed = raw.withColumn("amount", F.col("amount").cast("decimal(12,2)"))
    latest = dedupe_latest(typed, key="order_id", order_by="updated_at")
    write_table(latest, "shop.silver.orders", mode="overwrite")
    return latest


def build_daily_revenue(spark: SparkSession) -> DataFrame:
    """Aggregate silver orders into revenue per store per day, excluding cancellations."""
    orders = spark.read.table("shop.silver.orders")
    daily = (orders.filter(F.col("status") != "CANCELLED")
             .groupBy(F.to_date("created_at").alias("order_date"), "store_id")
             .agg(F.sum("amount").alias("revenue")))
    daily.write.mode("overwrite").saveAsTable("shop.gold.daily_revenue")
    return daily


class LateEventHandler:
    """Route events that arrive after the watermark to a quarantine table."""

    def __init__(self, spark: SparkSession, watermark_hours: int = 6):
        self.spark = spark
        self.watermark_hours = watermark_hours

    def split(self, events: DataFrame) -> tuple[DataFrame, DataFrame]:
        cutoff = F.current_timestamp() - F.expr(f"INTERVAL {self.watermark_hours} HOURS")
        on_time = events.filter(F.col("event_time") >= cutoff)
        late = events.filter(F.col("event_time") < cutoff)
        return on_time, late

    def quarantine(self, late: DataFrame) -> None:
        late.write.mode("append").saveAsTable("shop.quarantine.late_orders")


def rank_customers(orders: DataFrame) -> DataFrame:
    w = Window.partitionBy("customer_id").orderBy(F.col("amount").desc())
    return orders.withColumn("rank", F.row_number().over(w))


if __name__ == "__main__":
    session = SparkSession.builder.getOrCreate()
    load_orders_silver(session)
    build_daily_revenue(session)
