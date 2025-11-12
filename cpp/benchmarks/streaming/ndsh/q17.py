# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

    @staticmethod
    def q17(run_config: RunConfig) -> pl.LazyFrame:
        """Query 17."""
        lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
        part = get_data(run_config.dataset_path, "part", run_config.suffix)

        var1 = "Brand#23"
        var2 = "MED BOX"

        q1 = (
            part.filter(pl.col("p_brand") == var1)
            .filter(pl.col("p_container") == var2)
            .join(lineitem, how="left", left_on="p_partkey", right_on="l_partkey")
        )

        return (
            q1.group_by("p_partkey")
            .agg((0.2 * pl.col("l_quantity").mean()).alias("avg_quantity"))
            .select(pl.col("p_partkey").alias("key"), pl.col("avg_quantity"))
            .join(q1, left_on="key", right_on="p_partkey")
            .filter(pl.col("l_quantity") < pl.col("avg_quantity"))
            .select(
                (pl.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly")
            )
        )



# Physical Plan

# SELECT ('avg_yearly',) [1]
#   SELECT ('_______________1',) [1]
#     REPARTITION ('_______________0',) [1]
#       SELECT ('_______________0',) [11]
#         PROJECTION ('key', 'l_extendedprice', 'l_quantity', 'avg_quantity') [11]
#           FILTER ('key', 'avg_quantity', 'l_extendedprice', 'l_quantity') [11]
#             JOIN Inner ('key',) ('p_partkey',) ('key', 'avg_quantity', 'l_extendedprice', 'l_quantity') [11]
#               SELECT ('key', 'avg_quantity') [1]
#                 SELECT ('p_partkey', 'avg_quantity') [1]
#                   SELECT ('p_partkey', '____________1') [1]
#                     GROUPBY ('p_partkey',) ('p_partkey', '_____________0__mean_sum', '_____________1__mean_count') [1]
#                       REPARTITION ('p_partkey', '_____________0__mean_sum', '_____________1__mean_count') [1]
#                         GROUPBY ('p_partkey',) ('p_partkey', '_____________0__mean_sum', '_____________1__mean_count') [11]
#                           PROJECTION ('p_partkey', 'l_quantity') [11]
#                             CACHE ('p_partkey', 'l_quantity', 'l_extendedprice') [11]
#                               PROJECTION ('p_partkey', 'l_quantity', 'l_extendedprice') [11]
#                                 JOIN Left ('p_partkey',) ('l_partkey',) ('p_partkey', 'p_container', 'p_brand', 'l_quantity', 'l_extendedprice') [11]
#                                   SHUFFLE ('p_partkey', 'p_container', 'p_brand') [11]
#                                     SCAN PARQUET ('p_partkey', 'p_container', 'p_brand') [1]
#                                   SHUFFLE ('l_partkey', 'l_quantity', 'l_extendedprice') [11]
#                                     SCAN PARQUET ('l_partkey', 'l_quantity', 'l_extendedprice') [11]
#               PROJECTION ('p_partkey', 'l_extendedprice', 'l_quantity') [11]
#                 CACHE ('p_partkey', 'l_quantity', 'l_extendedprice') [11]
#                   PROJECTION ('p_partkey', 'l_quantity', 'l_extendedprice') [11]
#                     JOIN Left ('p_partkey',) ('l_partkey',) ('p_partkey', 'p_container', 'p_brand', 'l_quantity', 'l_extendedprice') [11]
#                       SHUFFLE ('p_partkey', 'p_container', 'p_brand') [11]
#                         SCAN PARQUET ('p_partkey', 'p_container', 'p_brand') [1]
#                       SHUFFLE ('l_partkey', 'l_quantity', 'l_extendedprice') [11]
#                         SCAN PARQUET ('l_partkey', 'l_quantity', 'l_extendedprice') [11]
