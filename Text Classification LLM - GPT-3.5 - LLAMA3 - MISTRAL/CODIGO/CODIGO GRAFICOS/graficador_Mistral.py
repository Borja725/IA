import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("resultados_mixtral.csv")

total=len(df["CODES"])
no_clasificado=len(df[df["PREDICTIONS"]=="[]"])
total_clasificado=total-no_clasificado
true_positive=len(df[df["COMPARISON"]==1])
false_positive=total_clasificado-true_positive

noClas_fp=len(df[df["COMPARISON"]==0])

labels = ['Total', 'Total Clasificado', 'No Clasificado', 'True Positive', 'False Positive']
values = [total, total_clasificado, no_clasificado, true_positive, false_positive]

# Crear el gráfico de barras
plt.figure(figsize=(5, 6))
plt.bar(labels, values, color=['blue', 'orange', 'purple', 'green', 'red', 'brown'])
plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Resultados del Clasificador')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()