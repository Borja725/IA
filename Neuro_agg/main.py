import pandas as pd
import logging
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import shutil
import sys

def fpri(informacion_mostrar):
    print(informacion_mostrar)
    logger.info(informacion_mostrar)
    print("")

def realizar_backup(origen, destino_zip):
    try:
        shutil.copytree(origen, destino_zip)
        
        shutil.make_archive(destino_zip, 'zip', destino_zip)
        
        print(f'Backup creado correctamente en {destino_zip}.zip')
    except Exception as e:
        print(f'Error al realizar el backup: {e}')

def enviar_correo(subject, body):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587 
    
    sender_email = 'bpellicer@odec.es'
    sender_password = 'Almoines2005+'
    
    receiver_email = 'bpellicer@odec.es'
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print('Correo electrónico enviado correctamente.')
    except Exception as e:
        print(f'Error al enviar correo electrónico: {e}')
    finally:

        server.quit()

logging.basicConfig(filename='informacion.log', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)


ruta_json = "informacion.json"
try:
    with open(ruta_json, 'r') as archivo:
        datos = json.load(archivo)
    logger.info("Archivo JSON cargado exitosamente.")
except FileNotFoundError:
    print("Archivo no encontrado")
    enviar_correo('Error en la ejecución del programa', f'Archivo no encontrado')
    sys.exit(1)

path1 = datos.get("path1")
path2 = datos.get("path2")

try:
    data = pd.read_excel(path1, sheet_name="sheet0")
except FileNotFoundError as e:
    logger.error(f"Archivo no encontrado: {e.filename}")
    enviar_correo('Error en la ejecución del programa', f'Archivo no encontrado')
    sys.exit(1)

logger.info(f"Datos de Excel cargados desde {path1}.")
    
data1 = pd.read_csv(path2)
logger.info(f"Datos CSV cargados desde {path2}.")
data1['True/False'] = data1['User'].isin(data['respid'])

filtered_data = data1[data1['True/False']]

fpri("data = pd.read_excel(file_path, sheet_name=sheet0)")
fpri("data1 = pd.read_csv(file_path2)")
fpri("data1['True/False'] = data1['User'].isin(data['respid'])")
fpri("filtered_data = data1[data1['True/False']]")
fpri("Leer los archivos introducidos y Creación columna True/False")
fpri(filtered_data)
print("-----------------------------------------------------------------------------------------------------")

campos = [
    'Veracity', 'V+', 'V-', 'Intensity', 'Interes', 'Rechazo',
    'Atributo', 'Compromiso', 'Confusion', 'Comprension', 'Performance'
]

average_func = {field: ['sum', 'mean'] for field in campos}

fpri("campos = [ 'Veracity', 'V+', 'V-', 'Intensity', 'Interes', 'Rechazo', 'Atributo', 'Compromiso', 'Confusion', 'Comprension', 'Performance']")
fpri("average_func = {field: ['sum', 'mean'] for field in campos}")
fpri("Creacion de dos subcampos para cada campo")
fpri(average_func)
print("-----------------------------------------------------------------------------------------------------")

grouped_data = filtered_data.groupby(['Module', 'User']).agg(average_func).reset_index()

fpri("grouped_data = filtered_data.groupby(['Module', 'User']).agg(average_func).reset_index()")
fpri("Agrupar cada fila por los campos Module y User y resetear el index")
fpri(grouped_data)
print("-----------------------------------------------------------------------------------------------------")

fpri("grouped_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_data.columns.values]")
fpri("mean_columns = [col for col in grouped_data.columns if col.endswith('_mean')]")
fpri("mean_columns = ['Module', 'User'] + mean_columns")
fpri("Guardar en un array las columnas")
print("-----------------------------------------------------------------------------------------------------")

filtered_grouped_data = grouped_data[~grouped_data['Module'].isin(['BBBB0001', 'BB0000'])]

fpri("filtered_grouped_data = grouped_data[~grouped_data['Module'].isin(['BBBB0001', 'BB0000'])][mean_columns].copy()")
fpri("filtered_grouped_data.sort_values(by='Module', inplace=True)")
fpri("Ordenar la columna Module por numeración")
fpri(filtered_grouped_data)
print("-----------------------------------------------------------------------------------------------------")

numeric_columns = filtered_grouped_data
filtered_grouped_data.loc[:, numeric_columns] = filtered_grouped_data.loc[:, numeric_columns].round(2)

fpri("numeric_columns = filtered_grouped_data.columns.difference(['Module', 'User'])")
fpri("filtered_grouped_data.loc[:, numeric_columns] = filtered_grouped_data.loc[:, numeric_columns].round(2)")
fpri("Establece nombre a cada columna y redondea los numeros a 2 decimales")
fpri(numeric_columns)
print("-----------------------------------------------------------------------------------------------------")

output_path_zipp = "C:\\Users\\bpellicer\\Downloads\\neuro_agg\\neuro_agg\\data_input"
output_path_zip = "C:\\Users\\bpellicer\\Downloads\\neuro_agg\\neuro_agg\\data_input\\zip_resultado"
output_path = "C:\\Users\\bpellicer\\Downloads\\neuro_agg\\neuro_agg\\data_input\\usuaris_agrupats_media_ordenats.csv"

filtered_grouped_data.to_csv(output_path, index=False)

fpri("output_path = 'C:\\Users\\bpellicer\\Downloads\\neuro_agg\\neuro_agg\\data_input\\usuaris_agrupats_media_ordenats.zip'")
fpri("filtered_grouped_data.to_csv(output_path, index=False)")
fpri("Creacion de variable con el path del numero archivo a crear y creación")
fpri(filtered_grouped_data)
enviar_correo('NEURO_AGG', f'Ejecución del programa Neuro_agg exitosa')
realizar_backup(output_path_zipp, output_path_zip)



