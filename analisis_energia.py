from pyspark.sql import SparkSession 
from pyspark.sql import functions as F 
from pyspark.sql.functions import ( 
col, lit, when, avg, first 
) 
from pyspark.sql.types import DoubleType 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
# 1. INICIALIZACIÓN DE SPARK 
spark = ( 
SparkSession.builder 
.appName("ClasificacionEnergiaEuropea_ProyectoPySpark") 
.config("spark.executor.memory", "8g") 
.getOrCreate() 
) 
# 2. LECTURA DEL DATASET DESDE ARCHIVO LOCAL 
base_path = os.path.dirname(os.path.abspath(__file__)) 
ruta_local = os.path.join(base_path, "data", "Temperaturas.csv") 
print(f"\nLeyendo archivo desde: {ruta_local}") 
df_ancho_original = ( 
spark.read 
.option("delimiter", ",") 
.option("header", True) 
.option("inferSchema", True) 
.csv(f"file://{ruta_local}") 
) 
print("\n--- Vista Preliminar del DataFrame Original ---") 
df_ancho_original.show(5, truncate=False) 
print(f"Columnas totales: {len(df_ancho_original.columns)}") 
df_ancho_original.printSchema() 
# 3. TRANSFORMACIÓN A FORMATO LARGO (PAÍS, TECNOLOGÍA) 
TECNOLOGIAS_REQUERIDAS = ["pv_national", "wind_national"] 
# Columnas de datos (todas menos 'time') 
columnas_datos = [c for c in df_ancho_original.columns if c.lower() != "time"] 
# Asumimos formato PAIS_TECNOLOGIA, ej: DE_pv_national 
paises_presentes = set( 
c.split("_")[0] 
for c in columnas_datos 
if len(c.split("_")) > 1 
) 
# Columnas que deberían existir 
columnas_requeridas = set( 
f"{pais}_{tec}" 
for pais in paises_presentes 
for tec in TECNOLOGIAS_REQUERIDAS 
) 
# Columnas faltantes 
columnas_faltantes = columnas_requeridas.difference(set(columnas_datos)) 
# Añadir columnas faltantes como NULL 
df_ancho_unificado = df_ancho_original 
for col_faltante in columnas_faltantes: 
df_ancho_unificado = df_ancho_unificado.withColumn( 
col_faltante, 
lit(None).cast(DoubleType()) 
) 
# Unpivot con stack: time, col_combinada, capacity_factor 
columnas_para_stack = [c for c in df_ancho_unificado.columns if c.lower() != 
"time"] 
num_columnas = len(columnas_para_stack) 
stack_pairs = ", ".join([f"'{c}', `{c}`" for c in columnas_para_stack]) 
stack_expression = f"{num_columnas}, {stack_pairs}" 
df_largo_analitico = ( 
df_ancho_unificado 
.selectExpr( 
"time", 
f"stack({stack_expression}) as (col_combinada, capacity_factor)" 
) 
.filter(col("capacity_factor").isNotNull()) 
) 
# Separar country (2 letras) y technology (resto) 
df_largo_analitico = ( 
df_largo_analitico 
.withColumn("country", F.substring(col("col_combinada"), 1, 2)) 
.withColumn( 
"technology", 
F.substring(col("col_combinada"), 4, F.length(col("col_combinada"))) 
) 
.select( 
"time", 
"country", 
"technology", 
col("capacity_factor").cast(DoubleType()).alias("capacity_factor") 
) 
) 
print("\n--- Vista Preliminar del DataFrame en Formato Largo (Analítico) ---") 
df_largo_analitico.show(5, truncate=False) 
# Extra: columna timestamp y mes (para procesamientos posteriores) 
df_largo_con_fecha = ( 
df_largo_analitico 
.withColumn("timestamp", F.to_timestamp("time")) 
.withColumn("year", F.year("timestamp")) 
.withColumn("month", F.month("timestamp")) 
) 
# 4. PROCESAMIENTO 1 
#    Promedios anuales Solar vs Eólico + Clasificación por país 
UMBRAL_MIXTURA = 0.10  # 10% 
# a) Agregado por país y tecnología (promedios) 
df_agregado_tec = df_largo_analitico.groupBy("country", "technology").agg( 
avg("capacity_factor").alias("avg_capacity_factor") 
) 
print("\nPromedios por país y tecnología") 
df_agregado_tec.show(10, truncate=False) 
# b) Pivot: una fila por país, columnas por tecnología 
df_agregado_pais = ( 
df_agregado_tec 
.groupBy("country") 
.pivot("technology") 
.agg(first("avg_capacity_factor")) 
) 
print("\n--- Después del pivot (df_agregado_pais) ---") 
df_agregado_pais.printSchema() 
df_agregado_pais.show(10, truncate=False) 
# c) Detección automática de columnas solar / eólica 
cols_agregado = df_agregado_pais.columns 
print("\nColumnas en df_agregado_pais:", cols_agregado) 
solar_candidates = [c for c in cols_agregado if "pv" in c.lower()] 
wind_candidates = [c for c in cols_agregado if "wind" in c.lower()] 
if not solar_candidates or not wind_candidates: 
raise Exception( 
f"No pude encontrar columnas de pv/wind después del pivot. " 
f"Columnas encontradas: {cols_agregado}" 
) 
solar_col = solar_candidates[0] 
wind_col = wind_candidates[0] 
print(f"\nUsando columna solar: {solar_col}") 
print(f"Usando columna eólica: {wind_col}") 
# d) Base para clasificación 
df_base_clasif = df_agregado_pais.select( 
"country", 
col(solar_col).alias("Solar_Avg"), 
col(wind_col).alias("Wind_Avg") 
) 
# e) Clasificación 
df_clasificacion_final = ( 
df_base_clasif 
.withColumn( 
"Clasificacion", 
when( 
(col("Solar_Avg").isNotNull()) & 
(col("Wind_Avg").isNotNull()) & 
(col("Solar_Avg") > col("Wind_Avg")) & 
((col("Solar_Avg") - col("Wind_Avg")) / col("Solar_Avg") > 
lit(UMBRAL_MIXTURA)), 
"Predominio Solar" 
) 
.when( 
(col("Solar_Avg").isNotNull()) & 
(col("Wind_Avg").isNotNull()) & 
(col("Wind_Avg") > col("Solar_Avg")) & 
((col("Wind_Avg") - col("Solar_Avg")) / col("Wind_Avg") > 
lit(UMBRAL_MIXTURA)), 
"Predominio Eólico" 
) 
.otherwise("Mixto o Balanceado") 
) 
.select("country", "Solar_Avg", "Wind_Avg", "Clasificacion") 
.sort("country") 
) 
print("\nClasificación Final por País") 
df_clasificacion_final.show(50, truncate=False) 
# ----- Gráficas P1 ----- 
df_pandas_clasificacion = df_clasificacion_final.toPandas() 
# Figura 1: Comparación de promedios Solar vs Eólico 
plt.figure(figsize=(18, 7)) 
paises = df_pandas_clasificacion["country"] 
solar = df_pandas_clasificacion["Solar_Avg"] 
wind = df_pandas_clasificacion["Wind_Avg"] 
bar_width = 0.35 
index = range(len(paises)) 
plt.bar( 
[i - bar_width / 2 for i in index], 
solar, 
bar_width, 
label="Factor Cap. Solar Promedio", 
color="#FFC300" 
) 
plt.bar( 
[i + bar_width / 2 for i in index], 
wind, 
bar_width, 
label="Factor Cap. Eólico Promedio", 
color="#581845" 
) 
plt.xlabel("País") 
plt.ylabel("Factor de Capacidad Promedio (0-1)") 
plt.title("Figura 1: Promedios Anuales Solar vs Eólico") 
plt.xticks(list(index), paises, rotation=60, ha="right") 
plt.legend() 
plt.grid(axis="y", linestyle="--") 
plt.tight_layout() 
plt.savefig("P1_promedios_solar_vs_eolico.png") 
# Figura 2: Conteo de países por clasificación 
plt.figure(figsize=(9, 6)) 
conteo_clasificacion = df_pandas_clasificacion["Clasificacion"].value_counts() 
conteo_clasificacion.plot(kind="bar", color=["#28B463", "#3498DB", "#F39C12"]) 
plt.title("Figura 2: Conteo de Países por Predominio Energético") 
plt.xlabel("Clasificación") 
plt.ylabel("Número de Países") 
plt.xticks(rotation=0) 
plt.tight_layout() 
plt.savefig("P1_conteo_clasificacion.png") 
# 5. PROCESAMIENTO 2 
#    Máximos y mínimos de factor de capacidad por país y tecnología 
df_maxmin = df_largo_analitico.groupBy("country", "technology").agg( 
F.max("capacity_factor").alias("max_capacity_factor"), 
F.min("capacity_factor").alias("min_capacity_factor") 
) 
print("\nMáximos y mínimos por país y tecnología") 
df_maxmin.show(20, truncate=False) 
# Pivot para tener columnas por tecnología (solo máximos para la gráfica) 
df_maxmin_pivot = ( 
df_maxmin 
.groupBy("country") 
.pivot("technology") 
.agg(first("max_capacity_factor")) 
) 
cols_max = df_maxmin_pivot.columns 
solar_max_candidates = [c for c in cols_max if "pv" in c.lower()] 
wind_max_candidates = [c for c in cols_max if "wind" in c.lower()] 
if solar_max_candidates and wind_max_candidates: 
solar_max_col = solar_max_candidates[0] 
wind_max_col = wind_max_candidates[0] 
df_maxmin_graph = df_maxmin_pivot.select( 
"country", 
col(solar_max_col).alias("Solar_Max"), 
col(wind_max_col).alias("Wind_Max") 
).sort("country") 
df_pandas_max = df_maxmin_graph.toPandas() 
plt.figure(figsize=(18, 7)) 
paises = df_pandas_max["country"] 
solar_max = df_pandas_max["Solar_Max"] 
wind_max = df_pandas_max["Wind_Max"] 
bar_width = 0.35 
index = range(len(paises)) 
plt.bar( 
[i - bar_width / 2 for i in index], 
solar_max, 
bar_width, 
label="Máx. Solar", 
color="#F1C40F" 
) 
plt.bar( 
[i + bar_width / 2 for i in index], 
wind_max, 
bar_width, 
label="Máx. Eólico", 
color="#2E86C1" 
) 
plt.xlabel("País") 
plt.ylabel("Máx. Factor de Capacidad (0-1)") 
plt.title("Figura 3: Máximos Anuales de Factor de Capacidad por País") 
plt.xticks(list(index), paises, rotation=60, ha="right") 
plt.legend() 
plt.grid(axis="y", linestyle="--") 
plt.tight_layout() 
plt.savefig("P2_maximos_solar_vs_eolico.png") 
# 6. PROCESAMIENTO 3 
#    Desviación estándar (volatilidad) por país y tecnología 
df_std = df_largo_analitico.groupBy("country", "technology").agg( 
F.stddev("capacity_factor").alias("std_capacity_factor") 
) 
print("\nDesviación estándar por país y tecnología") 
df_std.show(20, truncate=False) 
df_std_pivot = ( 
df_std 
.groupBy("country") 
.pivot("technology") 
.agg(first("std_capacity_factor")) 
) 
cols_std = df_std_pivot.columns 
solar_std_candidates = [c for c in cols_std if "pv" in c.lower()] 
wind_std_candidates = [c for c in cols_std if "wind" in c.lower()] 
if solar_std_candidates and wind_std_candidates: 
solar_std_col = solar_std_candidates[0] 
wind_std_col = wind_std_candidates[0] 
df_std_graph = df_std_pivot.select( 
"country", 
col(solar_std_col).alias("Solar_STD"), 
col(wind_std_col).alias("Wind_STD") 
).sort("country") 
df_pandas_std = df_std_graph.toPandas() 
plt.figure(figsize=(18, 7)) 
paises = df_pandas_std["country"] 
solar_std = df_pandas_std["Solar_STD"] 
wind_std = df_pandas_std["Wind_STD"] 
bar_width = 0.35 
index = range(len(paises)) 
plt.bar( 
[i - bar_width / 2 for i in index], 
solar_std, 
bar_width, 
label="STD Solar", 
color="#8E44AD" 
) 
plt.bar( 
[i + bar_width / 2 for i in index], 
wind_std, 
bar_width, 
label="STD Eólico", 
color="#16A085" 
) 
plt.xlabel("País") 
plt.ylabel("Desviación Estándar del Factor de Capacidad") 
plt.title("Figura 4: Volatilidad de Solar vs Eólico por País") 
plt.xticks(list(index), paises, rotation=60, ha="right") 
plt.legend() 
plt.grid(axis="y", linestyle="--") 
plt.tight_layout() 
plt.savefig("P3_std_solar_vs_eolico.png") 
# 7. PROCESAMIENTO 4 
#    Promedios mensuales por tecnología (perfil estacional) 
df_mensual = ( 
df_largo_con_fecha 
.groupBy("month", "technology") 
.agg( 
avg("capacity_factor").alias("avg_capacity_factor") 
) 
) 
print("\nPromedios mensuales por tecnología") 
df_mensual.orderBy("month", "technology").show(24, truncate=False) 
df_mensual_pivot = ( 
df_mensual 
.groupBy("month") 
.pivot("technology") 
.agg(first("avg_capacity_factor")) 
.orderBy("month") 
) 
cols_mens = df_mensual_pivot.columns 
solar_mens_candidates = [c for c in cols_mens if "pv" in c.lower()] 
wind_mens_candidates = [c for c in cols_mens if "wind" in c.lower()] 
if solar_mens_candidates and wind_mens_candidates: 
solar_mens_col = solar_mens_candidates[0] 
wind_mens_col = wind_mens_candidates[0] 
df_mens_graph = df_mensual_pivot.select( 
"month", 
col(solar_mens_col).alias("Solar_Mensual"), 
col(wind_mens_col).alias("Wind_Mensual") 
).orderBy("month") 
df_pandas_mens = df_mens_graph.toPandas() 
plt.figure(figsize=(10, 6)) 
meses = df_pandas_mens["month"] 
solar_mens = df_pandas_mens["Solar_Mensual"] 
wind_mens = df_pandas_mens["Wind_Mensual"] 
plt.plot(meses, solar_mens, marker="o", label="Solar", linestyle="-") 
plt.plot(meses, wind_mens, marker="s", label="Eólico", linestyle="--") 
plt.xlabel("Mes") 
plt.ylabel("Factor de Capacidad Promedio (0-1)") 
plt.title("Figura 5: Perfil Mensual Solar vs Eólico") 
plt.xticks(list(meses)) 
plt.grid(True, linestyle="--", alpha=0.6) 
plt.legend() 
plt.tight_layout() 
plt.savefig("P4_promedios_mensuales_solar_vs_eolico.png") 
# 8. PROCESAMIENTO 5 
#    Ranking de países según "score medio" (promedio Solar+Eólico) 
df_ranking = ( 
df_clasificacion_final 
.withColumn( 
"Score_Medio", 
(col("Solar_Avg") + col("Wind_Avg")) / 2.0 
) 
.orderBy(col("Score_Medio").desc()) 
) 
print("\nRanking de países por Score Medio") 
df_ranking.show(50, truncate=False) 
df_pandas_rank = df_ranking.toPandas() 
# Tomamos top 10 para la gráfica 
df_pandas_top10 = df_pandas_rank.head(10) 
plt.figure(figsize=(10, 6)) 
plt.bar( 
df_pandas_top10["country"], 
df_pandas_top10["Score_Medio"], 
color="#E67E22" 
) 
plt.xlabel("País") 
plt.ylabel("Score Medio (Promedio Solar + Eólico)") 
plt.title("Figura 6: Top 10 Países por Score Energético Medio") 
plt.grid(axis="y", linestyle="--", alpha=0.6) 
plt.tight_layout() 
plt.savefig("P5_ranking_top10_score_medio.png") 
# 9. CIERRE 
spark.stop() 
print("\nProceso de PySpark completado y sesión de Spark detenida.") 
print("Archivos de resultados guardados:") 
print(" - P1_promedios_solar_vs_eolico.png") 
print(" - P1_conteo_clasificacion.png") 
print(" - P2_maximos_solar_vs_eolico.png") 
print(" - P3_std_solar_vs_eolico.png") 
print(" - P4_promedios_mensuales_solar_vs_eolico.png") 
print(" - P5_ranking_top10_score_medio.png") 

