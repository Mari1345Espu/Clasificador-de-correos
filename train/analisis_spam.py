import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Cargar el conjunto de datos
try:
    df = pd.read_csv(r'C:\Users\mafee\Clasificador de correos\data\spam_dataset(1).csv')
except FileNotFoundError:
    print("Error: No encuentra el archivo dataset.")
    exit()

# Seleccionar los features que usaremos
features = ['suspicious_terms_count', 'subject_length', 'link_count', 'auth_pass_spf_dkim_dmarc']
X = df[features]
y = df['label']

# Mapear 'SPAM' a 1 y 'HAM' a 0
y = y.map({'SPAM': 1, 'HAM': 0})

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

print("\n--- Entrenamiento y evaluaciones completadas ---")

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

print("Gráfico del historial de métricas generado y guardado como 'Historico_metricas.png'.")

# --- Gráfica de la línea de regresión de la ÚLTIMA ÉPOCA ---
# Se utiliza el modelo y los datos de la última ejecución (i = num_executions - 1)
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

print("Gráfico de la línea de regresión final generado y guardado como 'Prediccion_final.png'.")