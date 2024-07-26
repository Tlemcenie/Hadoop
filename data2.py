from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, avg, explode, split, when, sum
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("FashionProductAnalysis").getOrCreate()

dataset_file = (spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("hdfs://0.0.0.0:9000/user/hadoop/hadoop/data_fashion.csv"))

products_by_brand = (dataset_file.groupBy("Brand")
                     .agg(count("*").alias("count"))
                     .orderBy(desc("count"))
                     .limit(10))

price_by_category = (dataset_file.groupBy("Category")
                     .agg(avg("Price").alias("avg_price"))
                     .orderBy(desc("avg_price"))
                     .limit(10))

products_by_season = (dataset_file.groupBy("Season")
                      .agg(count("*").alias("count"))
                      .orderBy(desc("count")))
avg_rating_by_brand = (dataset_file.groupBy("Brand")
                       .agg(avg("Rating").alias("avg_rating"))
                       .orderBy(desc("avg_rating"))
                       .limit(10))

common_style_attributes = (dataset_file.select(explode(split("Style Attributes", ",")).alias("attribute"))
                           .groupBy("attribute")
                           .count()
                           .orderBy(desc("count"))
                           .limit(10))

price_ranges = (dataset_file.select(
    when(col("Price") < 20, "0-20")
    .when((col("Price") >= 20) & (col("Price") < 50), "20-50")
    .when((col("Price") >= 50) & (col("Price") < 100), "50-100")
    .otherwise("100+")
    .alias("price_range"))
    .groupBy("price_range")
    .count()
    .orderBy("price_range"))

top_categories_by_reviews = (dataset_file.groupBy("Category")
                             .agg(sum("Review Count").alias("total_reviews"))
                             .orderBy(desc("total_reviews"))
                             .limit(10))

correlation = dataset_file.stat.corr("Price", "Rating")

common_sizes = (dataset_file.select(explode(split("Available Sizes", ",")).alias("size"))
                .groupBy("size")
                .count()
                .orderBy(desc("count"))
                .limit(10))

avg_price_by_rating = (dataset_file.groupBy("Rating")
                       .agg(avg("Price").alias("avg_price"))
                       .orderBy("Rating"))

brands = [row['Brand'] for row in products_by_brand.collect()]
brand_counts = [row['count'] for row in products_by_brand.collect()]

categories = [row['Category'] for row in price_by_category.collect()]
category_prices = [row['avg_price'] for row in price_by_category.collect()]

seasons = [row['Season'] for row in products_by_season.collect()]
season_counts = [row['count'] for row in products_by_season.collect()]

top_brands = [row['Brand'] for row in avg_rating_by_brand.collect()]
brand_ratings = [row['avg_rating'] for row in avg_rating_by_brand.collect()]

style_attributes = [row['attribute'] for row in common_style_attributes.collect()]
attribute_counts = [row['count'] for row in common_style_attributes.collect()]

price_range_labels = [row['price_range'] for row in price_ranges.collect()]
price_range_counts = [row['count'] for row in price_ranges.collect()]

top_review_categories = [row['Category'] for row in top_categories_by_reviews.collect()]
review_counts = [row['total_reviews'] for row in top_categories_by_reviews.collect()]

common_size_labels = [row['size'] for row in common_sizes.collect()]
size_counts = [row['count'] for row in common_sizes.collect()]

ratings = [row['Rating'] for row in avg_price_by_rating.collect()]
avg_prices = [row['avg_price'] for row in avg_price_by_rating.collect()]

plt.figure(figsize=(20, 20))

plt.subplot(331)
plt.bar(brands, brand_counts)
plt.title('Top 10 Brands by Product Count')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')

plt.subplot(332)
plt.bar(categories, category_prices)
plt.title('Top 10 Categories by Average Price')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Price')

plt.subplot(333)
plt.pie(season_counts, labels=seasons, autopct='%1.1f%%')
plt.title('Products by Season')

plt.subplot(334)
plt.bar(top_brands, brand_ratings)
plt.title('Top 10 Brands by Average Rating')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Rating')


plt.subplot(335)
plt.bar(style_attributes, attribute_counts)
plt.title('Top 10 Style Attributes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')

plt.subplot(336)
plt.bar(price_range_labels, price_range_counts)
plt.title('Product Distribution by Price Range')
plt.ylabel('Count')

plt.subplot(337)
plt.bar(top_review_categories, review_counts)
plt.title('Top 10 Categories by Review Count')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total Reviews')

plt.subplot(338)
plt.bar(common_size_labels, size_counts)
plt.title('Top 10 Available Sizes')
plt.ylabel('Count')

plt.subplot(339)
plt.plot(ratings, avg_prices, marker='o')
plt.title('Average Price by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Price')

plt.tight_layout()
plt.savefig('fashion_product_analysis_extended.png')
plt.close()

print("Visualization saved as 'fashion_product_analysis_extended.png'")
print(f"Correlation between Price and Rating: {correlation}")

spark.stop()