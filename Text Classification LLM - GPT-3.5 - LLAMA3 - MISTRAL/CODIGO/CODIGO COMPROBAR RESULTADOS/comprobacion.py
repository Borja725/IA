import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("verbatim_predictions_few_shot_V5.csv")

# Limpiar los datos: Convertir las cadenas de códigos a sets
df['CODES'] = df['CODES'].apply(lambda x: set(code.strip() for code in x.split(',')))

# Limpiar los datos: Convertir las cadenas de predicciones a sets
# Evalúa la cadena para convertirla en una lista y luego en un set
df['PREDICTIONS'] = df['PREDICTIONS'].apply(lambda x: set(eval(x)))

# Comparar CODES con PREDICTIONS y crear la nueva columna 'MATCH'
df['MATCH'] = df.apply(lambda row: 1 if row['PREDICTIONS'] == row['CODES'] else 0, axis=1)

# Contar el número de coincidencias
match_count = df['MATCH'].sum()

# Calcular el porcentaje de coincidencias
total_rows = df.shape[0]
accuracy_percentage = (match_count / total_rows) * 100

# Mostrar el DataFrame con la nueva columna y el conteo de coincidencias
print(df.head(50))
print(f'Número de coincidencias (MATCH=1): {match_count}')
print(f'Porcentaje de acierto: {accuracy_percentage:.2f}%')
