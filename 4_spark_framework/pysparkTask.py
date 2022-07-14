import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
from pyspark.sql.window import Window
from pyspark.sql.functions import *

appName = "PySpark PostgreSQL Example"
master = "local"
spark = SparkSession.builder.master(master).appName(appName).getOrCreate()
engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/postgres?client_encoding=utf8")

pdactor = pd.read_sql("select * from public.actor", engine)
pdfilmactor = pd.read_sql("select * from public.film_actor", engine)
pdfilm = pd.read_sql("select * from public.film", engine).dropna(axis="columns", how="all")
pdcategory = pd.read_sql("select * from public.category", engine)
pdfilmcategory = pd.read_sql("select * from public.film_category", engine)
pdsalesbyfilmcategory = pd.read_sql("select * from public.sales_by_film_category", engine)
pdinventory = pd.read_sql("select * from public.inventory", engine)
pdcity = pd.read_sql("select * from public.city", engine)
pdaddress = pd.read_sql("select * from public.address", engine)
pdcustomer = pd.read_sql("select * from public.customer", engine)
pdrental = pd.read_sql("select * from public.rental", engine)

# Convert Pandas dataframe to spark DataFrame
actor = spark.createDataFrame(pdactor)
filmactor = spark.createDataFrame(pdfilmactor)
film = spark.createDataFrame(pdfilm)
category = spark.createDataFrame(pdcategory)
filmcategory = spark.createDataFrame(pdfilmcategory)
salesbyfilmcategory = spark.createDataFrame(pdsalesbyfilmcategory)
inventory = spark.createDataFrame(pdinventory)
city = spark.createDataFrame(pdcity)
address = spark.createDataFrame(pdaddress)
customer = spark.createDataFrame(pdcustomer)
rental = spark.createDataFrame(pdrental)

# 1 Вывести количество фильмов в каждой категории, отсортировать по убыванию

filmcategory.join(category, ["category_id"]).groupBy("name").count().orderBy("count", ascending=False).show()

# 2 Вывести 10 актеров, чьи фильмы большего всего арендовали, отсортировать по убыванию

filmactor.join(actor, ["actor_id"]).groupBy("last_name", "first_name").count().orderBy("count", ascending=False).show(10)

# 3 Вывести категорию фильмов, на которую потратили больше всего денег

salesbyfilmcategory.groupBy("category").sum("total_sales").show(1)

# 4 Вывести названия фильмов, которых нет в inventory

film.join(inventory, ["film_id"], how='left').filter("inventory_id is NULL").select("title").show()

# 5 Вывести топ 3 актеров, которые больше всего появлялись в фильмах в категории “Children”. Если у нескольких актеров
# одинаковое кол-во фильмов, вывести всех.

windowPartition = Window.partitionBy("name").orderBy(col("count").desc())

actor.join(filmactor, ["actor_id"]).join(film, ["film_id"]).join(filmcategory, ["film_id"]).join(category, ["category_id"]). \
    filter(category.name == "Children").groupBy("name", "last_name", "first_name").count(). \
    withColumn("rank", dense_rank().over(windowPartition)).orderBy("rank").filter("rank <= 3"). \
    select("last_name", "first_name", "count").show()

# 6 Вывести города с количеством активных и неактивных клиентов (активный — customer.active = 1).
# Отсортировать по количеству неактивных клиентов по убыванию.

joined = customer.join(address, ["address_id"]).join(city, ["city_id"])

active = joined.filter("active = 1").groupBy("city").count().select(col("city"), col("count").alias("active_count"))
inactive = joined.filter("active = 0").groupBy("city").count().select(col("city"), col("count").alias("inactive_count"))

active.join(inactive, ["city"], how='full').fillna(value=0).orderBy("inactive_count", ascending=False).show(100)

# 7 Вывести категорию фильмов, у которой самое большое кол-во часов суммарной аренды в городах
# (customer.address_id в этом city), и которые начинаются на букву “a”. Тоже самое сделать для городов
# в которых есть символ “-”.

diffsec = rental.withColumn("DiffInSeconds", unix_timestamp("return_date") - unix_timestamp("rental_date"))
diffhours = diffsec.withColumn('duration', round(col('DiffInSeconds')/3600))

df = customer.join(address, ["address_id"]).join(city, ["city_id"]).join(diffhours, ["customer_id"]). \
    join(inventory, ["inventory_id"]).join(film, ["film_id"]).join(filmcategory, ["film_id"]).join(category, ["category_id"]). \
    groupBy("city", "name").sum("duration")

windowPart = Window.partitionBy("city").orderBy(col("sum(duration)").desc())

df.withColumn("rank", dense_rank().over(windowPart)).orderBy("rank").filter("rank = 1"). \
    filter(col("city").like("A%")).select("city", "name", "sum(duration)").show()

df.withColumn("rank", dense_rank().over(windowPart)).orderBy("rank").filter("rank = 1"). \
    filter(col("city").like("%-%")).select("city", "name", "sum(duration)").show()