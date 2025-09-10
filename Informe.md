
# Analisis de Calsificacion de SPAM

En este archivo se describira, el proceso de construccion y evaluacion de regresion logistica para clasificar el dataset creado en clase, de correos denominados como SPAM o HAM. 

# Regresion logistica Binaria

Esta me sirve para predecir la probabilidad de que una variable categorica dependienten tenga solo dos calores posibles, como si o no, usando una funcion sigmoide para transformar la salida en un valor entre 0 y 1, que puede interpreatrse como probabilidad.

# Explicacion del codigo

El analisis se baso en el archivo **spam_dataset (1).csv**. Los features que escogimos para entrenar el modelo, fueron solamente tres, entre ellas:
  -> Numero de terminos sospechosos
  -> El tamaño del asunto
  -> numero  de links
  -> Resultado de atentificacion

**Estructura del Código**
1. Importación de Librerías
```python 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
```

Importamos librerias para:
- Manipulación de datos (Pandas, NumPy)
- Machine Learning (Scikit-learn)
- Visualización (Matplotlib, Seaborn)
- Guardar historico de  de modelos (Joblib)

2. Carga y Preparación de Datos
```python
# Cargar el conjunto de datos
try:
    df = pd.read_csv(r'C:\Users\mafee\Clasificador de correos\data\spam_dataset(1).csv')
except FileNotFoundError:
    print("Error: No encuentra el archivo dataset.")
    exit()

# Seleccionar los features
features = ['suspicious_terms_count', 'subject_length', 'link_count', 'auth_pass_spf_dkim_dmarc']
X = df[features]
y = df['label']

# Mapear etiquetas a valores numéricos
y = y.map({'SPAM': 1, 'HAM': 0})
```
Aqui, leemos nuestro documento en la direccion que se encuentra el archivo,  y seleccionamos los features con los que trabajaremos o realizaremos el trabajo, que como los mencione anteriormente validan la gegitimidad de los remitentes y la veracidad del contenido de los correos.
Cargo el dataset con los 1000 correos electronicos existentes, y como clasificacion debo mapear la etiqueta: SPAM -> 1 y HAM -> 0. 
Por lo que se divide el entrenamiento en 70% y el 30% para testing.

3. Entrenamiento del modelo
``` python
# Lista para guardar el historial de métricas
history = []
num_executions = 5

# Bucle para realizar las épocas de entrenamiento
for i in range(num_executions):
    print(f"--- Ejecución {i+1}/{num_executions} ---")
    
    # División 70/30 con shuffling en cada iteración
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=i)
    
    # Entrenamiento del modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predicciones
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Exactitud: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Guardar el modelo y las métricas en el historial
    model_filename = f'logistic_regression_model_ejecutado{i+1}.joblib'
    joblib.dump(model, model_filename)
    
    history.append({
        'run': i + 1,
        'accuracy': accuracy,
        'f1_score': f1,
        'model_path': model_filename
    })
```

Aqui, entrenamos el modelo 5 veces, recordemos que usando la libreria **sklearning** este me entrega un modelo preciso desde la primera epocam por lo que entrenamos el algoritmo 5 veces realizando shuffling a los datos para evaluar la consistencia en las iteraciones y que nos de un algoritmo mas preciso, ademas de que se entrena el modelo, se evalua su precision a traves de los metodos de exactitud (acurracy) y F1 Score, ademas de que en cada historico de la ejecucion el modelo, se guarda para tener una comparacion. 

$$\text{Accuracy} = \frac{\text{Número de predicciones correctas}}{\text{Total de predicciones}}$$

**F1-score:** Media armonica entre la precision y re call, esto mas cuando se presenta desbalance en los numeros de caso de clasificacion.

4. Visualizacion de resultados
   
Aqui graficamos el historico de las metricas evaluadas y que tan constantes se mantuvieron durante las 5 ejecuciones.
```python
# Convertir el historial a un DataFrame
history_df = pd.DataFrame(history)

# Gráfica del historial de métricas
plt.figure(figsize=(10, 6))
plt.plot(history_df['run'], history_df['accuracy'], marker='o', label='Exactitud', color='blue')
plt.plot(history_df['run'], history_df['f1_score'], marker='o', label='F1-Score', color='red')
plt.title('Historial de Métricas del Modelo a lo Largo de las Ejecuciones')
plt.xlabel('Número de Ejecución (Época)')
plt.ylabel('Puntuación de la Métrica')
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid(True)
plt.savefig('Historico_metricas.png')
plt.close()
```
En este caso, podemos observar que, 

5. Visualizacion de la funcion sigmoide
6. Esta visualiza el modelo trasforma las caracteristicas en probabilidades usando la funcion sigmoide:
   
```python
# Gráfica de la línea de regresión de la ÚLTIMA ÉPOCA
final_model = model
X_test_final = X_test
y_test_final = y_test
probabilities_final = final_model.predict_proba(X_test_final)[:, 1]

plt.figure(figsize=(10, 6))

# Usamos 'subject_length' para visualizar la relación
subject_length_idx = features.index('subject_length')
coef = final_model.coef_[0][subject_length_idx]
intercept = final_model.intercept_[0]

x_plot = np.linspace(X_test_final['subject_length'].min(), X_test_final['subject_length'].max(), 300)
z_plot = intercept + coef * x_plot
p_plot = 1 / (1 + np.exp(-z_plot))

plt.plot(x_plot, p_plot, color='green', linewidth=2, label='Función Sigmoide del Modelo')
plt.axhline(0.5, linestyle='--', color='gray', label='Umbral de Decisión (0.5)')

sns.scatterplot(x=X_test_final['subject_length'], y=probabilities_final, hue=y_test_final.map({1: 'SPAM', 0: 'HAM'}),
                palette={'SPAM': 'red', 'HAM': 'blue'}, alpha=0.6, s=50)

plt.title('Función Sigmoide y Predicciones del Modelo Final')
plt.xlabel('Longitud del Asunto (subject_length)')
plt.ylabel('Probabilidad de ser SPAM')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.savefig('Prediccion_final.png')
plt.close()
```
Recordemos que el modelo de regresion logistica de la libreria usa la funcion sigmoide: 
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
para la combinacion de cada uno de los features que agregamos y los pondera, multiplicandolos con los coeficientes del modelo, en este caso solo se selecciono la variable subject_length, ya que no se puede graficar un modelo de mas de 2D.

## Conclusiones
 El modelo de regresion logistica binomial, ayuda a predecir y detectar el SPAM, digamos que se puede observar que el nivel de exatitudfue mayor al 09.99% en la mayoria de los casos ademas que el valor F1-Score se mantuvo con los mismos valores, confirmando que el modelo tiene un rendimiento consistente. 

 Ademas que el uso de la libreria de **train_test_split** para hacer shuffling durante el testeo, ayudo a validar que el modelo no dependa de un orden particular para 



