# Renewable Energy Analysis with PySpark

Proyecto de análisis de datos energéticos utilizando PySpark para procesar y clasificar países según su producción de energía renovable.

---

## 🎯 Objetivo

El objetivo del proyecto es analizar datos de generación de energía en distintos países y clasificar su perfil energético en función del predominio de fuentes renovables como:

- Energía solar ☀️  
- Energía eólica 🌬️  
- Generación mixta ⚡  

Se busca identificar patrones y tendencias utilizando herramientas de Big Data.

---

## 🧠 Enfoque del proyecto

El análisis se realizó utilizando PySpark para manejar grandes volúmenes de datos y aplicar transformaciones eficientes.

El flujo del proyecto incluye:

1. Carga de datos  
2. Limpieza y preprocesamiento  
3. Transformaciones  
4. Análisis exploratorio  
5. Clasificación de países  

---

## ⚙️ Tecnologías utilizadas

- Python  
- PySpark  
- Procesamiento distribuido  
- DataFrames  

---

## 📂 Archivos del proyecto

- `analisis_energia.py` → script principal donde se realiza todo el análisis  
- `BDNRPROYFIN.pdf` → reporte completo con metodología, resultados y conclusiones  

---

## 🔍 Metodología

El análisis se basa en:

- Uso de DataFrames para manipulación de datos  
- Aplicación de filtros y agregaciones  
- Cálculo de métricas relevantes  
- Clasificación basada en criterios definidos  

---

## 💻 Ejemplo de código

A continuación se muestra un fragmento del procesamiento con PySpark, donde se carga el dataset y se transforma a un formato analítico:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, avg
from pyspark.sql.types import DoubleType

spark = (
    SparkSession.builder
    .appName("ClasificacionEnergiaEuropea_ProyectoPySpark")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

df_ancho_original = (
    spark.read
    .option("delimiter", ",")
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/Temperaturas.csv")
)

df_largo = (
    df_ancho_original
    .selectExpr(
        "time",
        "stack(2, 'solar', pv_national, 'wind', wind_national) as (technology, capacity_factor)"
    )
    .filter(col("capacity_factor").isNotNull())
)
```

Este fragmento muestra la lectura del dataset con Spark, la transformación de formato ancho a largo y la preparación del DataFrame para análisis por tecnología.

## 📊 Resultados

- Identificación de países con alta dependencia en energía renovable  
- Diferenciación entre modelos energéticos  
- Detección de patrones en producción energética  

---

## 🚀 Cómo ejecutar

1. Asegúrate de tener PySpark instalado  

2. Ejecuta el script:

```bash
python analisis_energia.py
```

🧩 Notas
El proyecto puede requerir configuración de Spark dependiendo del entorno
Los resultados completos y visualizaciones se encuentran en el PDF
👨‍💻 Autor

Julian Quiroz
ITAM – Finanzas y Ciencia de Datos