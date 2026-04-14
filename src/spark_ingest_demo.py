"""PySpark ingestion demo — 10 K mocked records covering almost all Spark SQL types.

Pipeline
--------
1. Generate ~10 K rows in Python covering: LongType, IntegerType, ShortType,
   ByteType, FloatType, DoubleType, DecimalType, StringType, BooleanType,
   DateType, TimestampType, ArrayType(String), StructType (nested), MapType.
2. Create a Spark DataFrame and register it as the SQL view  →  raw_users
3. Transform via Spark SQL + DataFrame API  →  transformed_users view
4. Write the flattened, CSV-safe columns to  /workspace/output/spark_demo_csv/
"""

import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RECORDS = 10_000
OUTPUT_DIR = Path("/workspace/output/spark_demo_csv")
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Schema — covers almost every Spark primitive + complex type
# ---------------------------------------------------------------------------
ADDRESS_SCHEMA = StructType(
    [
        StructField("street", StringType(), True),
        StructField("city", StringType(), True),
        StructField("zip_code", StringType(), True),
    ]
)

SCHEMA = StructType(
    [
        StructField("id", LongType(), False),  # 64-bit int
        StructField("name", StringType(), True),  # text
        StructField("age", IntegerType(), True),  # 32-bit int
        StructField("score", DoubleType(), True),  # 64-bit float
        StructField("rating", FloatType(), True),  # 32-bit float
        StructField("balance", DecimalType(15, 2), True),  # exact decimal
        StructField("is_active", BooleanType(), True),  # bool
        StructField("birth_date", DateType(), True),  # calendar date
        StructField("created_at", TimestampType(), True),  # timestamp
        StructField("tags", ArrayType(StringType()), True),  # string array
        StructField("address", ADDRESS_SCHEMA, True),  # nested struct
        StructField(
            "metadata", MapType(StringType(), StringType()), True
        ),  # string map
        StructField("category_id", ShortType(), True),  # 16-bit int
        StructField("byte_flag", ByteType(), True),  # 8-bit int
    ]
)

# ---------------------------------------------------------------------------
# Lookup tables for realistic-looking mock values
# ---------------------------------------------------------------------------
_TAGS = ["electronics", "clothing", "food", "sports", "home", "books", "toys", "beauty"]
_CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "Dallas",
]
_STREETS = [
    "Main St",
    "Oak Ave",
    "Maple Dr",
    "Cedar Ln",
    "Pine Rd",
    "Elm Blvd",
    "Washington St",
    "Park Ave",
]

_START_DATE = date(1960, 1, 1)
_END_DATE = date(2005, 12, 31)
_DATE_RANGE = (_END_DATE - _START_DATE).days

_START_TS = datetime(2020, 1, 1)
_END_TS = datetime(2026, 4, 13)
_TS_RANGE = int((_END_TS - _START_TS).total_seconds())


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------
def _rand_date(rng: random.Random) -> date:
    return _START_DATE + timedelta(days=rng.randint(0, _DATE_RANGE))


def _rand_ts(rng: random.Random) -> datetime:
    return _START_TS + timedelta(seconds=rng.randint(0, _TS_RANGE))


def generate_rows(n: int, seed: int = RANDOM_SEED) -> list[tuple]:
    """Return *n* tuples whose positions match SCHEMA field order."""
    rng = random.Random(seed)
    rows: list[tuple] = []
    for i in range(n):
        # Nested struct — plain dict matches ADDRESS_SCHEMA fields
        address = {
            "street": rng.choice(_STREETS),
            "city": rng.choice(_CITIES),
            "zip_code": str(rng.randint(10000, 99999)),
        }
        # Map — 1-3 arbitrary key/value pairs
        metadata = {
            f"key_{j}": f"val_{rng.randint(1, 100)}" for j in range(rng.randint(1, 3))
        }
        rows.append(
            (
                i + 1,  # id          LongType
                f"user_{i + 1:05d}",  # name        StringType
                rng.randint(18, 80),  # age         IntegerType
                round(rng.uniform(0.0, 100.0), 6),  # score       DoubleType
                float(round(rng.uniform(1.0, 5.0), 2)),  # rating      FloatType
                Decimal(
                    str(round(rng.uniform(-10_000.0, 50_000.0), 2))
                ),  # balance     DecimalType(15,2)
                rng.choice([True, False]),  # is_active   BooleanType
                _rand_date(rng),  # birth_date  DateType
                _rand_ts(rng),  # created_at  TimestampType
                rng.sample(_TAGS, rng.randint(1, 4)),  # tags        ArrayType(String)
                address,  # address     StructType
                metadata,  # metadata    MapType(String,String)
                rng.randint(1, 100),  # category_id ShortType
                rng.randint(0, 127),  # byte_flag   ByteType
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    spark = (
        SparkSession.builder.appName("spark-ingest-demo")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # Step 1 — build raw DataFrame
    # ------------------------------------------------------------------
    print(f"\n[1/4]  Generating {NUM_RECORDS:,} mock rows …")
    rows = generate_rows(NUM_RECORDS)
    raw_df = spark.createDataFrame(rows, schema=SCHEMA)

    # ------------------------------------------------------------------
    # Step 2 — register first view
    # ------------------------------------------------------------------
    raw_df.createOrReplaceTempView("raw_users")
    print("[2/4]  Spark SQL view created: raw_users")
    raw_df.printSchema()

    # ------------------------------------------------------------------
    # Step 3 — transformations (mix of SQL and DataFrame API)
    # ------------------------------------------------------------------
    print("[3/4]  Applying transformations …")

    # SQL-based transformations:
    #   • flatten address struct into top-level columns
    #   • derive age_bucket from age
    #   • derive score_category from score
    #   • compute days_since_birth
    #   • compute tag_count and join tags into a string
    #   • extract map keys as an array
    #   • filter: keep active users or those with score > 60
    transformed_df = spark.sql(
        """
        SELECT
            id,
            name,
            age,
            CASE
                WHEN age BETWEEN 18 AND 30 THEN 'young'
                WHEN age BETWEEN 31 AND 50 THEN 'middle'
                ELSE                             'senior'
            END                                             AS age_bucket,
            ROUND(score, 2)                                 AS score,
            CASE
                WHEN score <  33.33 THEN 'low'
                WHEN score <  66.67 THEN 'medium'
                ELSE                     'high'
            END                                             AS score_category,
            rating,
            CAST(balance AS DOUBLE)                         AS balance,
            is_active,
            birth_date,
            DATEDIFF(CURRENT_DATE(), birth_date)            AS days_since_birth,
            DATE_FORMAT(created_at, 'yyyy-MM-dd HH:mm:ss') AS created_at,
            SIZE(tags)                                      AS tag_count,
            ARRAY_JOIN(tags, ', ')                          AS tags_str,
            address.street                                  AS street,
            address.city                                    AS city,
            address.zip_code                                AS zip_code,
            MAP_KEYS(metadata)                              AS metadata_keys,
            category_id,
            byte_flag
        FROM raw_users
        WHERE is_active = TRUE
           OR score > 60.0
        """
    )

    # DataFrame API transformation: flag high-value accounts
    transformed_df = transformed_df.withColumn(
        "high_value",
        F.when(
            (F.col("balance") > 30_000) & (F.col("score_category") == "high"),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )

    # ------------------------------------------------------------------
    # Step 4 — register second view
    # ------------------------------------------------------------------
    transformed_df.createOrReplaceTempView("transformed_users")
    print("[4/4]  Spark SQL view created: transformed_users")
    transformed_df.printSchema()

    # ------------------------------------------------------------------
    # Quick summary via SQL on the new view
    # ------------------------------------------------------------------
    print("\n── Age-bucket / score-category distribution ──")
    spark.sql(
        """
        SELECT
            age_bucket,
            score_category,
            COUNT(*)                    AS record_count,
            ROUND(AVG(balance), 2)      AS avg_balance,
            SUM(CAST(high_value AS INT)) AS high_value_count
        FROM transformed_users
        GROUP BY age_bucket, score_category
        ORDER BY age_bucket, score_category
        """
    ).show()

    print("── Top cities by average score ──")
    spark.sql(
        """
        SELECT
            city,
            COUNT(*)               AS users,
            ROUND(AVG(score), 2)   AS avg_score,
            ROUND(AVG(balance), 2) AS avg_balance
        FROM transformed_users
        GROUP BY city
        ORDER BY avg_score DESC
        """
    ).show()

    # ------------------------------------------------------------------
    # Write to CSV (flat, CSV-safe columns only)
    # ------------------------------------------------------------------
    csv_cols = [
        "id",
        "name",
        "age",
        "age_bucket",
        "score",
        "score_category",
        "rating",
        "balance",
        "is_active",
        "birth_date",
        "days_since_birth",
        "created_at",
        "tag_count",
        "tags_str",
        "street",
        "city",
        "zip_code",
        "category_id",
        "byte_flag",
        "high_value",
    ]
    output_path = str(OUTPUT_DIR)
    print(f"\nWriting CSV → {output_path}")
    (
        transformed_df.select(*csv_cols)
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(output_path)
    )

    total = transformed_df.count()
    print(f"Wrote {total:,} rows to CSV.  Done.\n")
    spark.stop()


if __name__ == "__main__":
    main()
