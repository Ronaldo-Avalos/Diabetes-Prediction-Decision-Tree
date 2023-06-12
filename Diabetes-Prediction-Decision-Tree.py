 # Arbol de Descición(Regresión)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv('diabetes.csv')

# Dividir los datos en variables independientes (X) y variable de salida (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier()

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Obtener la profundidad del árbol
depth = model.get_depth()
print("Profundidad del árbol:", depth)

# Obtener el número de hojas o nodos terminales
num_leaves = model.get_n_leaves()
print("Número de hojas:", num_leaves)

# Evaluar la precisión del modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Visualizar el árbol de decisión
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()
