import pandas as pd
import ast

# Ruta al archivo CSV
file_path = 'D:\\LLMs_Clasificador\\archivo_modificado.csv'

# Leer el CSV en un DataFrame de pandas
df = pd.read_csv(file_path)

# Función para formatear los códigos a cuatro dígitos y comparar con PREDICTIONS
def compare_codes_predictions(row):
    codes = row['CODES']  # Obtener el valor de la columna CODES
    predictions = ast.literal_eval(row['PREDICTIONS'])  # Convertir la cadena a lista de manera segura

    # Formatear los códigos a enteros eliminando los ceros a la izquierda
    formatted_codes = [str(int(code)) for code in codes.split(',')]

    # Convertir predictions a una lista de códigos eliminando los ceros iniciales
    formatted_predictions = [str(int(pred)) for pred in predictions]

    # Comparar los conjuntos de códigos formateados
    if set(formatted_codes) == set(formatted_predictions):
        return '1'
    else:
        return '0'

# Crear una nueva columna 'COMPARISON_RESULT' con los resultados de la comparación
df['COMPARISON_RESULT'] = df.apply(compare_codes_predictions, axis=1)

# # Guardar el DataFrame modificado en un nuevo archivo CSV en la carpeta actual
# output_file_path = './resultados18.csv'
# df.to_csv(output_file_path, index=False)

# print(f"Se ha guardado el archivo CSV con la columna de comparación en: {output_file_path}")

# Calcular el porcentaje de filas en las que COMPARISON_RESULT es 'IGUAL'
total_rows = len(df)
igual_count = df[df['COMPARISON_RESULT'] == '1'].shape[0]
porcentaje_igual = (igual_count / total_rows) * 100

# Imprimir el porcentaje por consola
print(f"Porcentaje de filas donde los códigos son iguales: {porcentaje_igual:.2f}%")

# Función para guardar las columnas VERBATIM, CODES, PREDICTIONS y COMPARISON formateadas en un archivo CSV
def save_formatted_codes_predictions(df, output_path):
    formatted_data = []
    for _, row in df.iterrows():
        codes = row['CODES']
        predictions = eval(row['PREDICTIONS'])

        # Formatear los códigos a cuatro dígitos con ceros a la izquierda si es necesario
        formatted_codes = [f'{int(code):04d}' for code in codes.split(',')]
        formatted_predictions = [f'{int(pred):04d}' for pred in predictions]

        formatted_data.append({
            'VERBATIM': row['VERBATIM'],
            'CODES': ','.join(formatted_codes),
            'PREDICTIONS': ','.join(formatted_predictions),
            'COMPARISON': '1' if set(formatted_codes) == set(formatted_predictions) else '0'
        })

    # Crear un DataFrame con los datos formateados
    formatted_df = pd.DataFrame(formatted_data)

    # Guardar el DataFrame formateado en un archivo CSV
    formatted_df.to_csv(output_path, index=False)

# Guardar las columnas formateadas en un nuevo archivo CSV
formatted_output_file_path = './resultados_formateados_Mistral.csv'
save_formatted_codes_predictions(df, formatted_output_file_path)

print(f"Se ha guardado el archivo CSV con las columnas formateadas en: {formatted_output_file_path}")
