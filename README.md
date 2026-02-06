# -Solicitudes-a-un-sitio-web
Predicción y análisis de solicitudes web a un único sitio web

Desarrollo de Modelo Predictivo para Análisis de Solicitudes en un Sitio Web
Creación de un modelo de aprendizaje automático para predecir y clasificar las solicitudes realizadas en un único sitio web, optimizando la gestión de datos y mejorando la toma de decisiones.
Tecnologías utilizadas: python, numpy, pandas, matplotlib, seaborn, plotly, datetime, sklearn, random, contextmanager, time, tqdm, lightgbm, tensorflow e imblearn.
Visualización: matplotlib, seaborn, plotly.expresspara análisis exploratorio de datos con gráficos dinámicos e interactivos.
Preprocesamiento: scikit-learn, LabelEncoder, category_encoders, SMOTEpara tratamiento de desequilibrios y codificación de variables categóricas.
Modelado: RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, LightGBM, integrados con estrategias como GridSearchCVy validación cruzada con KFoldpara optimizar parámetros y mejorar la generalización.
Métricas de evaluación: accuracy_score, mean_squared_error, log_losspara medir rendimiento y precisión del modelo.
Frameworks adicionales: TensorFlowpara la creación de modelos de redes neuronales en casos específicos.

**Resultados clave:**
Implementación de un pipeline eficiente de análisis y predicción, reduciendo el error absoluto medio en un X% y mejorando la precisión hasta X%.
Visualización avanzada de tendencias y resultados para facilitar la interpretación de las predicciones.
Habilidades aplicadas: Machine Learning, análisis exploratorio de datos (EDA), optimización de modelos, gestión de desequilibrios en datasets y trabajo con grandes volúmenes de datos.

🌐 Predicción de Tráfico Web con Machine Learning

Este proyecto aborda el análisis y la predicción del tráfico web a partir de un conjunto de datos de series temporales, utilizando técnicas de análisis exploratorio (EDA) y un modelo de Gradient Boosting basado en LightGBM.

El objetivo principal es modelar y predecir el volumen de solicitudes web (TrafficCount) a partir de características temporales extraídas del timestamp, evaluando el desempeño mediante validación cruzada.

🎯 Objetivos del proyecto

Analizar el comportamiento histórico del tráfico web.

Explorar relaciones entre variables temporales.

Transformar datos de series temporales en variables predictivas.

Entrenar un modelo de regresión robusto para predicción de tráfico.

Evaluar desempeño mediante métricas de error y visualizaciones.

📊 Dataset

Fuente: Registro de solicitudes web a un único sitio

Tipo: Serie temporal

Variable objetivo: TrafficCount

Variable temporal: Timestamp

Variables derivadas

A partir del timestamp se generan:

Year

Month

Day

Hour

📌 El dataset no presenta valores faltantes ni duplicados relevantes.

🔍 Metodología
1️⃣ Análisis Exploratorio de Datos (EDA)

Inspección de estructura y estadísticos descriptivos.

Histogramas de variables numéricas.

Análisis de correlaciones mediante heatmap.

Evolución del tráfico a lo largo del tiempo.

Identificación de horas pico de tráfico.

2️⃣ Ingeniería de Características

Conversión de Timestamp a formato datetime.

Extracción de variables temporales (año, mes, día, hora).

Codificación de variables categóricas mediante LabelEncoder.

Selección de variables numéricas relevantes.

3️⃣ Preparación de Datos

Separación manual de datos en entrenamiento (80%) y prueba (20%).

Construcción de un pipeline de generación de features.

Conversión a matrices numéricas para el modelado.

4️⃣ Modelado Predictivo
📌 Modelo utilizado

LightGBM Regressor

📌 Estrategia de entrenamiento

Validación cruzada K-Fold (5 folds).

Optimización con función de pérdida RMSE.

Uso de early stopping implícito vía evaluación por fold.

📌 Principales hiperparámetros

learning_rate: 0.1

max_depth: 5

num_leaves: 62

n_estimators: 10.000

subsample: 0.9

colsample_bytree: 0.5

5️⃣ Evaluación del Modelo

Métrica principal: Root Mean Squared Error (RMSE).

Comparación entre valores reales y predichos.

Distribución de predicciones en entrenamiento y prueba.

Gráfico de dispersión: valores reales vs predichos.

6️⃣ Interpretabilidad

Análisis de importancia de variables utilizando:

Importancia por ganancia (gain) de LightGBM.

Visualización de las features más relevantes.

Evaluación del impacto de la variable Hour en el tráfico.

📈 Resultados

El modelo captura correctamente los patrones temporales del tráfico web.

Las variables horarias muestran alta influencia en la predicción.

LightGBM ofrece buen balance entre rendimiento y capacidad predictiva.

El uso de validación cruzada reduce el riesgo de overfitting.

📌 Este enfoque es adecuado para:

Planeamiento de capacidad

Optimización de infraestructura

Análisis de comportamiento de usuarios

🛠️ Tecnologías y Librerías

Python

Pandas / NumPy

Matplotlib / Seaborn

Plotly

Scikit-learn

LightGBM

TQDM

📁 Estructura del proyecto
├── web_traffic.csv
├── 1.py
└── README.md
▶️ Cómo ejecutar el proyecto

Clonar el repositorio:

git clone https://github.com/tu-usuario/web-traffic-prediction.git

Instalar dependencias:

pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm tqdm

Ejecutar el script:

python 1.py
🚀 Posibles mejoras futuras

Incorporar variables exógenas (eventos, campañas, feriados).

Modelos específicos de series temporales (LSTM, Prophet).

Feature engineering cíclico (seno/coseno para hora y mes).

Early stopping explícito.

Deploy del modelo como API para predicción en tiempo real.

👤 Autor

Flavia Hepp
Proyecto de Data Science aplicado a series temporales y analítica web.
