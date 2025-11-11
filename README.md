#Тестовое задание

## Общее время: 6 часов 30 минут

    🚀 30 минут - настройка облачной инфраструктуры
    📊 6 часов - анализ и обработка данных

## Инфраструктура
Облачная платформа

    Провайдер: Yandex Cloud
    Сервис: DataProc (управляемый Apache Spark)
    Технология: PySpark 3.0.3

Конфигурация кластера

    Executor instances: 2
    Executor memory: 5530m
    Доступно ядер: 2
    Мастер URL: yarn

Датасет

Источник: 100 Million Data CSV (https://www.kaggle.com/datasets/zanjibar/100-million-data-csv/data)

Описание: Торговые данные Японии за период 1988-2020 годов

    Файл: custom_1988_2020.csv
    Размер: 113,607,321 записей
    Содержание: Данные по экспорту/импорту, товарные коды, страны, стоимость в йенах

## Код
### Импорт библиотек и настройка

```python
%pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import concat, lpad, lit
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка Spark для оптимальной производительности
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "400")

bucket_name = 'sparkwrk'

df = spark.read.option("header", True).csv(f's3a://{bucket_name}/bi_datasets/custom_1988_2020.csv')
df.printSchema()
```
![](/screen/1.png)

### Проверка и предобработка данных

```python
%pyspark

initial_partitions = df.rdd.getNumPartitions()

print(f"Исходное количество партиций: {initial_partitions}")

print("Оригинальные колонки:")
all_columns = df.columns
for i, col_name in enumerate(all_columns):
    print(f"   {i}: '{col_name}'")

print(f"\nВсего колонок: {len(all_columns)}")

rename_dict = {}
for col_name in all_columns:
    if "198801" in str(col_name) or col_name == "198801":
        rename_dict[col_name] = "ym"
    elif col_name == "1":
        rename_dict[col_name] = "exp_imp"  
    elif col_name == "103":
        rename_dict[col_name] = "hs9"
    elif col_name == "100":
        rename_dict[col_name] = "customs"
    elif "000000190" in str(col_name) or col_name == "000000190":
        rename_dict[col_name] = "country"
    elif col_name == "0":
        rename_dict[col_name] = "q1"
    elif col_name == "35843":
        rename_dict[col_name] = "q2"  
    elif col_name == "34353":
        rename_dict[col_name] = "value_yen"

print(f"\nСоответствия:")

for old_name, new_name in rename_dict.items():
    print(f"   '{old_name}' -> '{new_name}'")

df_processed = df.select([col(c).alias(rename_dict.get(c, c)) for c in df.columns])


if initial_partitions * 2 > 200:
    optimal_partitions = initial_partitions * 2
else:
    optimal_partitions = 200

df_processed = df_processed.repartition(optimal_partitions)
print(f"Оптимизированное количество партиций: {df_processed.rdd.getNumPartitions()}")

# Приводим типы данных
if "ym" in df_processed.columns:
    df_processed = df_processed.withColumn("ym", col("ym").cast(IntegerType()))
    df_processed = df_processed.withColumn("year", (col("ym") / 100).cast(IntegerType()))
    df_processed = df_processed.withColumn("month", when((col("ym") % 100) < 10, concat(lit("0"), (col("ym") % 100).cast(StringType()))).otherwise((col("ym") % 100).cast(StringType())))

if "exp_imp" in df_processed.columns:
    df_processed = df_processed.withColumn("exp_imp", col("exp_imp").cast(StringType()))

if "hs9" in df_processed.columns:
    df_processed = df_processed.withColumn("hs9", col("hs9").cast(StringType()))

if "customs" in df_processed.columns:
    df_processed = df_processed.withColumn("customs", col("customs").cast(IntegerType()))

if "country" in df_processed.columns:
    df_processed = df_processed.withColumn("country", col("country").cast(StringType()))

if "q1" in df_processed.columns:
    df_processed = df_processed.withColumn("q1", col("q1").cast(LongType()))

if "q2" in df_processed.columns:
    df_processed = df_processed.withColumn("q2", col("q2").cast(LongType()))

if "value_yen" in df_processed.columns:
    df_processed = df_processed.withColumn("value_yen", col("value_yen").cast(LongType()))

df_processed.cache()

print("Схема после обработки:")
df_processed.printSchema()

print("Образец данных:")
available_cols = [col for col in ["year", "month", "exp_imp", "hs9", "country", "q1", "q2", "value_yen"] if col in df_processed.columns]
df_processed.select(*available_cols).show(5)
```
![](/screen/2.png)

![](/screen/3.png)

![](/screen/4.png)

![](/screen/5.png)

### Очистка данных

```python
%pyspark


filters = []

if "year" in df_processed.columns and "month" in df_processed.columns:
    filters.append(col("year").isNotNull() & col("month").isNotNull())

if "exp_imp" in df_processed.columns:
    filters.append(col("exp_imp").isNotNull() & (col("exp_imp") != ""))

if "hs9" in df_processed.columns:
    filters.append(col("hs9").isNotNull() & (col("hs9") != ""))

if "country" in df_processed.columns:
    filters.append(col("country").isNotNull() & (col("country") != ""))

if "value_yen" in df_processed.columns:
    filters.append(col("value_yen").isNotNull() & (col("value_yen") > 0))

if "q1" in df_processed.columns:
    filters.append(col("q1").isNotNull())

if "q2" in df_processed.columns:
    filters.append(col("q2").isNotNull())

# Объединяем все фильтры
combined_filter = filters[0]
for filter_condition in filters[1:]:
    combined_filter = combined_filter & filter_condition

df_clean = df_processed.filter(combined_filter)
print(f"После удаления пустых: {df_clean.count():,} записей")

# Фильтруем строки с цифрами в hs9
if "hs9" in df_clean.columns:
    df_with_digits = df_clean.filter(col("hs9").rlike(".*[0-9].*"))
    print(f"После фильтрации строк с цифрами: {df_with_digits.count():,} записей")
else:
    df_with_digits = df_clean
    print("Колонка hs9 не найдена")

# Поиск дублей
key_fields = [field for field in ["year", "month", "exp_imp", "hs9", "country", "customs"] if field in df_with_digits.columns]
print(f"Ключевые поля для дедупликации: {key_fields}")

if len(key_fields) > 0:
    if "country" in key_fields:
        df_partitioned = df_with_digits.repartition("country")
        print("Данные перепартиционированы по country")
    else:
        df_partitioned = df_with_digits
    
    print("Анализ дублей с параллелизмом...")
    duplicates = df_partitioned.groupBy(*key_fields).count().filter(col("count") > 1).cache()
    
    duplicate_count = duplicates.count()
    total_duplicate_records = duplicates.agg(sum("count")).collect()[0][0] if duplicate_count > 0 else 0
    
    print(f"Найдено групп дублей: {duplicate_count:,}")
    print(f"Всего дублированных записей: {total_duplicate_records:,}")
    
    if duplicate_count > 0:
        print("Топ-10 дублей:")
        duplicates.orderBy(desc("count")).show(10)
    
    df_deduplicated = df_partitioned.dropDuplicates(key_fields)
else:
    df_deduplicated = df_with_digits

final_count = df_deduplicated.count()
print(f"Финальный датасет: {final_count:,} записей")
print(f"Финальное количество партиций: {df_deduplicated.rdd.getNumPartitions()}")

df_deduplicated.cache()
```
![](/screen/6.png)

### Агрегация по времени
```python
%pyspark


if "year" in df_deduplicated.columns and "month" in df_deduplicated.columns:
    
    df_time_partitioned = df_deduplicated.repartition("year", "month")
    print(f"Данные перепартиционированы по времени: {df_time_partitioned.rdd.getNumPartitions()} партиций")
    
    agg_expressions = []
    
    if "exp_imp" in df_deduplicated.columns:
        agg_expressions.append(countDistinct("exp_imp").alias("unique_exp_imp"))
    
    if "hs9" in df_deduplicated.columns:
        agg_expressions.append(countDistinct("hs9").alias("unique_hs9"))
    
    if "country" in df_deduplicated.columns:
        agg_expressions.append(countDistinct("country").alias("unique_countries"))
    
    numeric_cols = ["customs", "q1", "q2", "value_yen"]
    for col_name in numeric_cols:
        if col_name in df_deduplicated.columns:
            agg_expressions.extend([
                avg(col_name).alias(f"avg_{col_name}"),
                expr(f"percentile_approx({col_name}, 0.5)").alias(f"median_{col_name}")
            ])
    
    agg_expressions.append(count("*").alias("total_records"))
    
    time_aggregation = df_time_partitioned.groupBy("year", "month").agg(*agg_expressions).orderBy("year", "month").cache()
    
    print("Агрегация по времени:")
    time_aggregation.show(50)
    
    print(f"Результат агрегации: {time_aggregation.count():,} записей")
    print(f"Партиций в результате: {time_aggregation.rdd.getNumPartitions()}")
    
else:
    print("Временные колонки не найдены")
    time_aggregation = None
```
![](/screen/7_1.png)
![](/screen/7_2.png)

### Аналитические метрики для числовой колонки

```python
%pyspark


numeric_column = "value_yen"

if numeric_column in df_with_metrics.columns:
    
    numeric_data = df_with_metrics.select(numeric_column).filter(col(numeric_column).isNotNull()).cache()
    
    print(f"\nАнализ колонки: {numeric_column}")
    print(f"Количество записей: {numeric_data.count():,}")
    
    # 1. Гистограмма - подготовка данных
    
    histogram_sample = numeric_data.sample(0.01, seed=42).cache()
    sample_count = histogram_sample.count()
    print(f"\nРазмер выборки для гистограммы: {sample_count:,}")
    
    hist_stats = histogram_sample.agg(
        min(numeric_column).alias("min_val"),
        max(numeric_column).alias("max_val"),
        avg(numeric_column).alias("mean_val"),
        stddev(numeric_column).alias("std_val")
    ).collect()[0]
    
    print(f"Диапазон: [{hist_stats['min_val']:,}, {hist_stats['max_val']:,}]")
    print(f"Среднее: {hist_stats['mean_val']:,.2f}")
    print(f"Стандартное отклонение: {hist_stats['std_val']:,.2f}")
    
    hist_data = histogram_sample.rdd.map(lambda row: float(row[0])).histogram(50)
    bins = hist_data[0]
    frequencies = hist_data[1]
    
    print(f"Гистограмма создана: {len(bins)-1} интервалов")
    print("Первые 10 интервалов и частот:")
    
    num_to_show = 10
    if len(frequencies) < 10:
        num_to_show = len(frequencies)
    
    for i in range(num_to_show):
        print(f"   [{bins[i]:,.0f}, {bins[i+1]:,.0f}): {frequencies[i]:,}")
    
    # 2. 95% доверительный интервал
    
    print(f"\n2. 95% доверительный интервал для {numeric_column}:")
    
    full_stats = numeric_data.agg(
        count(numeric_column).alias("n"),
        avg(numeric_column).alias("mean"),
        stddev(numeric_column).alias("stddev")
    ).collect()[0]
    
    n = full_stats["n"]
    mean_val = full_stats["mean"]
    std_val = full_stats["stddev"]
    
    print(f"Полная выборка: n={n:,}, mean={mean_val:.2f}, std={std_val:.2f}")
    
    
    print(f"\nМетодика расчета доверительного интервала:")
    print(f"Размер выборки: n={n:,}")
    
    if n >= 30:
        print("n >= 30")
        print("Используем нормальное распределение с z-критическим значением")
        print("Для 95%: z = 1.96")
        
        z_critical = 1.96
        margin_error = z_critical * (std_val / math.sqrt(n))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        method = "ЦПТ + нормальное распределение"
        
    else:
        print("n < 30 ") 
        print("df = n-1")
        print("4. Для 95%: t-критическое ≈ 2.0")
        
        t_critical = 2.0
        margin_error = t_critical * (std_val / math.sqrt(n))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        method = "t-распределение"
    
    print(f"\nРезультат 95% доверительного интервала:")
    print(f"   Метод: {method}")
    print(f"   Интервал: [{ci_lower:,.2f}, {ci_upper:,.2f}]")
    print(f"   Ширина интервала: {ci_upper - ci_lower:,.2f}")
    print(f"   Погрешность: ±{margin_error:,.2f}")
    
    print(f"\nС вероятностью 95% истинное среднее значение {numeric_column}")
    print(f"находится в интервале [{ci_lower:,.2f}, {ci_upper:,.2f}]")
```
![](/screen/8_1.png)

![](/screen/8_2.png)

![](/screen/8_3.png)

### График среднего значения денежэных вложений в Йенах (тыс.) по месяцам

```python
if time_aggregation is not None and f"avg_{numeric_column}" in time_aggregation.columns:
    
    # Подготавливаем данные для графика
    monthly_avg_data = time_aggregation.select("year", "month", f"avg_{numeric_column}", "total_records").orderBy("year", "month")
    
    
    chart_data = monthly_avg_data.withColumn("period", 
                                            concat(col("year").cast("string"), 
                                                  lit("-"), 
                                                  lpad(col("month"), 2, "0")))
    
    # Собираем данные
    rows = chart_data.collect()
    
    print("%table period\taverage_value\ttotal_records")
    for row in rows:
        period = row.period
        avg_val = getattr(row, f"avg_{numeric_column}")
        total = row.total_records
        print(f"{period}\t{avg_val:.2f}\t{total}")
    
    print(f"\nПодготовлено {len(rows)} точек для графика")

else:
    print("Данные для графика среднего значения не найдены")
```
![](/screen/8_4.png)

### Merge метрики

```python
%pyspark

if time_aggregation is not None:
    df_for_join = df_deduplicated.repartition("year", "month")
    time_agg_for_join = time_aggregation.repartition("year", "month")
    
    print("Данные перепартиционированы для оптимального join")
    
    df_with_metrics = df_for_join.join(time_agg_for_join, ["year", "month"], "left").cache()
    
    print(f"Датасет с метриками: {df_with_metrics.count():,} записей")
    print(f"Партиций после join: {df_with_metrics.rdd.getNumPartitions()}")
    
    print("Образец данных с метриками:")
    sample_cols = [col for col in ["year", "month", "value_yen", "unique_countries", "avg_value_yen"] if col in df_with_metrics.columns]
    df_with_metrics.select(*sample_cols).show(10)
    
    null_metrics_count = df_with_metrics.filter(col("total_records").isNull()).count()
    print(f"Записей без метрик: {null_metrics_count:,}")
    
else:
    df_with_metrics = df_deduplicated
    print("Используем данные без временных метрик")
```

![](/screen/9.png)

### Случайное разделение датасета на 3 части

```python
%pyspark


df_sample1, df_sample2, df_sample3 = df_with_metrics.randomSplit([0.25, 0.25, 0.5], seed=42)

df_sample1.cache()
df_sample2.cache() 
df_sample3.cache()

count1 = df_sample1.count()
count2 = df_sample2.count()
count3 = df_sample3.count()
total = count1 + count2 + count3

print(f"РАЗДЕЛЕНИЕ ДАТАСЕТА:")
print(f"   Выборка 1: {count1:,} записей ({count1/total*100:.1f}%)")
print(f"   Выборка 2: {count2:,} записей ({count2/total*100:.1f}%)")
print(f"   Выборка 3: {count3:,} записей ({count3/total*100:.1f}%)")
print(f"   Всего: {total:,} записей")

print(f"Партиции в выборках:")
print(f"   Выборка 1: {df_sample1.rdd.getNumPartitions()} партиций")
print(f"   Выборка 2: {df_sample2.rdd.getNumPartitions()} партиций")
print(f"   Выборка 3: {df_sample3.rdd.getNumPartitions()} партиций")

print("\nРаспределение по годам в выборках:")
for i, df_sample in enumerate([df_sample1, df_sample2, df_sample3], 1):
    if "year" in df_sample.columns:
        year_dist = df_sample.groupBy("year").count().orderBy("year")
        print(f"\nВыборка {i} - топ-10 годов:")
        year_dist.show(10)
```
![](/screen/10.png)

### Доп.задание 2

```python
%pyspark

# Данные конкурента
competitor_successes = 5
competitor_failures = 995

# Данные компании
our_failures = 200

# Априорное распределение на основе данных конкурента
alpha_prior = competitor_successes
beta_prior = competitor_failures

# Апостериорное распределение после наших наблюдений
alpha_posterior = alpha_prior + 0 
beta_posterior = beta_prior + our_failures

# Вероятность успеха следующего прототипа
probability_success = alpha_posterior / (alpha_posterior + beta_posterior)

print(f"Вероятность успеха 201-го прототипа: {probability_success:.4f} или {probability_success*100:.2f}%")

# Доверительный интервал
beta_dist = stats.beta(alpha_posterior, beta_posterior)
ci_lower, ci_upper = beta_dist.interval(0.95)
print(f"95% доверительный интервал: [{ci_lower:.4f}, {ci_upper:.4f}]")
```
![](/screen/11.png)



























