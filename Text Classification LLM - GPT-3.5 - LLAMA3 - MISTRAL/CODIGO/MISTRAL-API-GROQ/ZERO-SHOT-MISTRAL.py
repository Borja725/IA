import os
import pandas as pd
from tqdm import tqdm
from groq import Groq

df = pd.read_csv("verbatims_nov_2023_Version_17072024.csv")

verbatims = df['VERBATIM'].to_numpy()
codes = df['CODES'].to_numpy()

client = Groq(
    api_key="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",
)

def classify_text(prompt, max_tokens=80):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def assign_tags(text):
    prompt = f"""
Clasifica el siguiente comentario en la etiqueta correspondiente o etiquetas correspondientes según el siguiente esquema:

"0101" = "El producto o la caja han llegado dañados o el paquete abierto.",
"0102" = "Ha habido un retraso en la entrega.",
"0103" = "Ha habido un error en el pedido (producto equivocado, productos de más, falta producto).",
"0104" = "Mensajeria: Ha habido fallos debidos a la empresa de repartidores, transporte (SEUR,Amazon,...). No se han enviado mensajes",
"0201" = "Comentarios negativos sobre los precios altos de los productos. Productos caros",
"0202" = "Faltan promociones, ofertas, regalos, Que me devuelvan el dinero. Mala gestión de las garantías, Quitar los gastos de envío.",
"0203" = "Publicidad engañosa, no han hecho lo que prometieron",
"0204" = "No Premiar fidelidad, Tratar mal a los clientes.",
"0205" = "Habla sobre el plan de suscripción de Nespresso.",
"0206" = "Faltan variedades de café, Falta de stock, Falta descafeinado, Mala calidad del café.",
"0207" = "Mala calidad de las máquinas. Fallos en las máquinas",
"0208" = "Habla negativamente sobre el producto pero no se puede codificar en ninguna de las anteriores.",
"0301" = "Puntos de Reciclaje. No recogen las cápsulas usadas, Que den bolsas para reciclar.",
"0302" = "Preocupación sobre el impacto ambiental y la sostenibilidad.",
"0401" = "El producto ha sido reparado por el servicio técnico y todavía no funciona tras la entrega.",
"0402" = "Tiempo de recogida mayor de lo estipulado para la recogida del producto.",
"0403" = "Tiempo de entrega lento de la máquina reparada.",
"0404" = "Lentitud en la comunicación con el servicio técnico.",
"0405" = "Problema con la máquina de sustitución.",
"0406" = "Personal del Servicio Tecnico no es bueno.",
"0495" = "Habla sobre el servicio técnico pero no se puede codificar en ninguna de las anteriores.",
"0501" = "Problema con la página web o con el pago online.",
"0601" = "Mala atención al cliente por parte de terceros (Amazon, el Corte Inglés, …).",
"0701" = "Mala atención al cliente por parte de Nespresso en la tienda física.",
"0702" = "Pocas muestras gratuitas. Problema en el proceso de degustación del producto en la tienda física.",
"0708" = "Falta de tiendas.",
"0808" = "Servicio de atención al cliente por llamada: Falta de soluciones, personal ineficaz, problemas de gestión de incidencias.",
"0095" = "Otros problemas que no se han codificado anteriormente.",
"9101" = "Satisfacción en la entrega o en la recepción. Buenas entregas",
"9201" = "Comentario positivo sobre la calidad del producto. Productos muy buenos. Buena variedad",
"9202" = "Buena estrategia de publicidad de la marca.",
"9203" = "Satisfacción con los beneficios por fidelidad.",
"9301" = "Satisfacción por el reciclaje y impacto ambiental.",
"9401" = "Satisfacción por el servicio técnico.",
"9501" = "Satisfacción por la página web.",
"9601" = "Satisfacción por la atención al cliente por parte de terceros en tiendas (Amazon, el Corte Inglés, …).",
"9701" = "Satisfacción por la atención al cliente por parte de Nespresso en tiendas.",
"9702" = "Satisfacción por el proceso de degustación del producto.",
"9801" = "Atención: Comentario positivo sobre la atención recibida."
"9995" = "Otros comentarios positivos que no se han codificado anteriormente.",
"0009" = "Nada. Contestaciones sin sentido".

Responde únicamente con la etiqueta correspondeinte, o si es el caso con las etiquetas correspondientes separadas por espacio.
No quiero la explicación de porque seleccionas cada etiqueta.
Si no estás seguro, devuelve únicamente "NO_PRED".

    Text: "{text}"

    Tags:
    """
    
    result = classify_text(prompt, max_tokens=40) 
    return [tag.strip() for tag in result.split(",")]

results=[]

# Test the function
for verb,code in tqdm(zip(verbatims, codes), total=len(verbatims)):
    article = verb
    tags = assign_tags(article)
    results.append({"VERBATIM": verb, "CODES": code, "PREDICTIONS": tags})

df_results = pd.DataFrame(results)

# Guardar en un archivo CSV
df_results.to_csv('verbatim_predictions_few_shot_V5.csv', index=False)