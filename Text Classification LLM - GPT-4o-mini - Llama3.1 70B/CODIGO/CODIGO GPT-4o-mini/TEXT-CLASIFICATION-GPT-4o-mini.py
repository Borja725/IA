import pandas as pd
import openai
import json
from tqdm import tqdm

openai.api_key =""

df = pd.read_csv("C:\\Users\\dcanet\\Downloads\\pruebas 10xCategoria\\Dataset_400.csv")

prompt_template = """
Dado el siguiente plan de codificación con el siguiente json de definición:

```json
{
"rules": {
        "negative_verbatims": {
            "Entregas - 01": {
                "description": "Comentarios negativos relacionados con la entrega de productos comprados o pedidos, excluyendo la entrega de máquinas reparadas.",
                "0101": {
                    "description": "El producto o la caja/paquete han llegado dañados en el momento de la entrega.",
                    "conditions": ["dañado", "roto", "golpeado", "mal empaquetado"]
                },
                "0102": {
                    "description": "Ha habido un retraso en la entrega. Excluye retrasos en la recogida.",
                    "conditions": ["retraso", "tardanza", "demora"]
                },
                "0103": {
                    "description": "Ha habido un error en el pedido (producto equivocado, productos de más, falta producto).",
                    "conditions": ["error en el contenido", "pedido equivocado", "faltan productos", "productos de más"]
                },
                "0104": {
                    "description": "Fallos debidos a la empresa de logística (SEUR, MRW …). Retraso por culpa de la empresa de transporte",
                    "conditions": ["mensajería", "logística", "transporte", "SEUR", "el repartidor no entregó el paquete", "el paquete no fue entregado", "pésimo servicio a domicilio"]
                    "examples": ["mala gestión por el servicio de transporte", "los de Correo siempre llegan tarde"]
                }
            },
            "Mkt & Promociones - 02": {
                "description": "Comentarios negativos relacionados con marketing y promociones.",
                "0201": {
                    "description": "Habla sobre el precio de los productos.",
                    "conditions": ["precio", "coste", "caro", "barato"]
                },
                "0202": {
                    "description": "Que me cambien la cafetera por una nueva, Faltan promociones, ofertas, regalos, Que me devuelvan el dinero, Que informen de las promociones, que no caduquen, Mala gestión de las garantías, Quitar los gastos de envío, Reparaciones demasiado caras.",
                    "conditions": ["cambiar", "promociones", "ofertas", "regalos", "dinero", "informen", "garantías", "envío", "reparaciones", "regalos por menos capsulas"]
                    "examples": ["como cambiar de máquina"]
                },
                "0203": {
                    "description": "Publicidad engañosa, decepción sobre la publicidad mostrada.",
                    "conditions": ["publicidad", "engañoso", "decepción"]
                },
                "0204": {
                    "description": "Quejas explícitas sobre la falta de reconocimiento o premios por fidelidad.",
                    "conditions": ["fidelidad no premiada", "fidelidad no reconocida", "El texto debe contener referencias a regalos, descuentos o tratos especiales por reconocimiento de su fidelidad a la marca"]
                    "examples": ["Sin detalles para los clientes antiguos"]
                },
                "0205": {
                    "description": "Habla sobre el plan de suscripción de Nespresso.",
                    "conditions": ["plan de suscripción", "suscripción"]
                },
                "0206": {
                    "description": "faltan variedades de cafe, falta de stock de capsulas de café, mala calidad del cafe. Cambios en el café. Disponibilidad del café.",
                    "conditions": ["variedades", "stock", "descafeinado", "calidad", "falta disponibilidad"]
                    "examples": ["Me encantaría que el café Pumpkin Spice estuviese disponible todo el año"]
                },
                "0207": {
                    "description": "Habla sobre las máquinas (más tiempo en el servicio técnico que en casa, mala calidad).",
                    "conditions": ["máquinas de mala calidad", "Es recurrente que la máquina se rompa o deje de funcionar"]
                    "examples": ["Tengo una máquina que simplemente no hace café, es frustrante."]
                },
                "0208": {
                    "description": "Habla sobre el producto pero no se puede codificar en ninguna de las anteriores.",
                    "conditions": ["producto", "temperatura del café", "confianza en el producto"]
                    "examples": ["Me gustaría que el café saliese más caliente", ]
                }
            },
            "Reciclaje - 03": {
                "description": "Comentarios negativos relacionados con reciclaje y sostenibilidad.",
                "0301": {
                    "description": "No recogen las cápsulas usadas, Que den bolsas para reciclar.",
                    "conditions": ["reciclaje", "cápsulas", "bolsas"]
                    "examples": ["Un servicio de suscripción para reciclar cápsulas podría ser muy efectivo."]
                },
                "0302": {
                    "description": "Preocupación sobre el impacto ambiental y la sostenibilidad.",
                    "conditions": ["sostenibilidad", "impacto ambiental"]
                }
            },
            "Servicio Técnico - 04": {
                "description": "Comentarios negativos relacionados con el servicio técnico, incluyendo la entrega de máquinas reparadas.",
                "0401": {
                    "description": "La reparación del producto no se completó, ya que el servicio técnico me lo devolvió en el mismo estado defectuoso en el que fue enviado, incumpliendo con la garantía de un arreglo funcional y duradero",
                    "conditions": ["Solo aplica si el servicio técnico devuelve el producto sin haber solucionado el problema"]
                    "examples": ["Tras ser reparada, mi cafetera aún tiene problemas con el filtro."]
                },
                "0402": {
                    "description": "Tiempo de recogida mayor de lo estipulado para la recogida del producto.",
                    "conditions": ["Recogida de máquina/producto defectuoso", "cambio de cafetera"]
                    "examples": ["que se lleven la máquina para arreglarla"]
                },
                "0403": {
                    "description": "Tiempo de entrega de la máquina reparada mayor de lo esperado.",
                    "conditions": ["Entrega de máquina reparada"]
                    "examples": ["todavía estoy esperando la máquina"]
                },
                "0404": {
                    "description": "Lentitud en la comunicación con el servicio técnico.",
                    "conditions": ["comunicación", "lentitud"]
                },
                "0405": {
                    "description": "Problema con la máquina de sustitución.",
                    "conditions": ["sustitución", "máquina", "Hace referencia únicamente a la máquina proporcionada como sustitución de la máquina del cliente que tiene la incidencia"]
                    "examples": ["decepción porque la máquina de cortesia no funcionaba"]
                },
                "0406": {
                    "description": "Problema con el servicio técnico en general sin justificación del motivo",
                    "conditions": ["Cuando el cliente expresamente indica que el servicio técnico no es bueno sin justificar el motivo", "mal servicio técnico en general"]
                }
            },
            "Website - 05": {
                "description": "Comentarios negativos relacionados con la página web o con el pago online.",
                "0501": {
                    "description": "Problema con la página web o con el pago online. Problema con la app o con el chat.",
                    "conditions": ["página web", "pago online", "app", "chat"]
                    "examples": ["el servicio online es pésimo", "por el chat no son claros"]
                }
            },
            "Trade - 06": {
                "description": "Comentarios negativos relacionados con la atención al cliente por parte de terceros.",
                "0601": {
                    "description": "Mala atención al cliente por parte de terceros (Amazon, el Corte Inglés, …).",
                    "conditions": ["Mala atención/servicio Amazon, Corte inglés o terceros"]
                }
            },
            "BTQ - 07": {
                "description": "Comentarios negativos relacionados con la atención al cliente en la tienda física de Nespresso.",
                "0701": {
                    "description": "Mala atención al cliente por parte de Nespresso en la tienda física.",
                    "conditions": ["atención al cliente", "tienda física"]
                },
                "0702": {
                    "description": "Problema en el proceso de degustación del producto en la tienda física. Ofrecer cafés.",
                    "conditions": ["degustación", "tienda física"]
                    "examples": ["volver a ofrecer café a los clientes"]
                },
                "0708": {
                    "description": "Falta de tiendas.",
                    "conditions": ["falta de tiendas", "renovación de tienda", "nuevas ubicaciones de tiendas", "nuevos puntos de recogidas"]
                }
            },
            "CRC - 08": {
                "description": "Comentarios negativos relacionados con el servicio de atención al cliente por llamada. Ninguna solución.",
                "0808": {
                    "description": "Falta de soluciones, Personal ineficaz, No saben gestionar las incidencias, Dicen que ya te llamarán y no lo hacen, Confusión con datos personales. Errores con otras personas",
                    "conditions": ["soluciones", "personal", "gestionar", "llamar", "datos personales", "mala respuesta"]
                    "examples": ["deben mejorar el servicio", "tener un servicio de atención al cliente profesional", "que se pongan en contacto conmigo", "siguen sin resolverme el problema", "ser más serios", "esperaba más"]
                }
            },
            "Otros - 95": {
                "description": "Otros comentarios negativos que no se han codificado anteriormente.",
                "95": {
                    "description": "Otros comentarios negativos no categorizados en las anteriores.",
                    "conditions": ["otros", "malos", "no volveré", "café caducado"]
                    "examples": ["no tengo la cafetera", "estoy decepcionado", "falta de educación", "incluir algún detalle tipo unos bombones, una taza… por tantas molestias]
                }
            }
        },
        "positive_verbatims": {
            "Entregas - 91": {
                "description": "Comentarios positivos relacionados con la entrega de productos comprados o pedidos, excluyendo la entrega de máquinas reparadas.",
                "9101": {
                    "description": "Satisfacción en la entrega del producto.",
                    "conditions": ["satisfacción", "entrega"]
                }
            },
            "Mkt & Promociones - 92": {
                "description": "Comentarios positivos relacionados con marketing y promociones. Calidad del producto",
                "9201": {
                    "description": "Satisfacción la marca o con los productos de la marca (café, máquinas, cápsulas, complementos cafetera). Buen sabor",
                    "conditions": ["satisfacción", "producto", "marca", "sabor"]
                },
                "9202": {
                    "description": "Buena estrategia de publicidad de la marca.",
                    "conditions": ["publicidad", "estrategia"]
                    "examples": ["Un lema bien elaborado puede llegar a convertirse en sinónimo de una marca."]
                },
                "9203": {
                    "description": "Satisfacción con los beneficios por fidelidad.",
                    "conditions": ["fidelidad", "beneficios"]
                    "examples": ["Valoro mucho los gestos de agradecimineto de Nespresso.", "Me gustan los obsequios en forma de cápsulas", "Adoro las ofertas exclusivas"]
                }
            },
            "Reciclaje - 93": {
                "description": "Comentarios positivos relacionados con reciclaje y sostenibilidad.",
                "9301": {
                    "description": "Satisfacción por el impacto ambiental.",
                    "conditions": ["satisfacción", "impacto ambiental"]
                }
            },
            "Servicio Técnico - 94": {
                "description": "Comentarios positivos relacionados con el servicio técnico, incluyendo la entrega de máquinas reparadas.",
                "9401": {
                    "description": "Satisfacción por el servicio técnico.",
                    "conditions": ["satisfacción", "servicio técnico"]
                }
            },
            "Website - 95": {
                "description": "Comentarios positivos relacionados con la página web.",
                "9501": {
                    "description": "Satisfacción por la página web.",
                    "conditions": ["satisfacción", "página web"]
                }
            },
            "Trade - 96": {
                "description": "Comentarios positivos relacionados con la atención al cliente por parte de terceros (Amazon, el Corte Inglés, MediaMarkt, …).",
                "9601": {
                    "description": "Satisfacción por la atención al cliente por parte de terceros (Amazon, el Corte Inglés, …).",
                    "conditions": ["satisfacción", "atención al cliente", "terceros"]
                }
            },
            "BTQ - 97": {
                "description": "Comentarios positivos relacionados con la atención al cliente únicamente en la tienda física de Nespresso. Excluye comentarios que no especifiquen en la tienda física",
                "9701": {
                    "description": "Satisfacción por la atención al cliente por parte de Nespresso en la tienda física. Excluye comentarios que no especifiquen en la tienda física",
                    "conditions": ["satisfacción", "atención al cliente", "tienda física"]
                    "examples": ["buena atención en la boutique de Madrid", "me resolvieron las dudas en la tienda de Valencia"]
                },
                "9702": {
                    "description": "Satisfacción por el proceso de degustación del producto. Buen sabor",
                    "conditions": ["satisfacción", "degustación", "sabor"]
                }
            },
            "CRC - 98": {
                "description": "Comentarios positivos relacionados con el servicio de atención al cliente. Respuesta satisfactoria. ",
                "9801": {
                    "description": "Satisfacción por la atención recibida. Profesionalidad. Valoración del problema. Atención por llamada",
                    "conditions": ["satisfacción", "atención", "profesionalidad"]
                    "examples": ["una persona de Nespresso me atendió de forma muy profesional", "Valoro mucho el trato y la facilidad", "atención de 10", "la consultora me solución los problemas", "buena atención recibida", "he llamado y me resolvieron la duda"]
                }
            },
            "Otros - 9995": {
                "description": "Otros comentarios positivos que no se han codificado anteriormente.",
                "9995": {
                    "description": "Otros comentarios positivos no categorizados en las anteriores.",
                    "conditions": ["otros", "masterclass", "gracias", "todo bien", "amabilidad con los clientes"]
                    "examples": ["seguir así", "todo es mejorable", "gracias", "adquirí gratis la cafetera"]
                }
            },
            "Otros - 9": {
                "description": "Otros comentarios que no se han codificado anteriormente.",
                "9": {
                    "description": "Otros comentarios no categorizados en las anteriores.",
                    "conditions": ["otros", "no", "nada", "no merece la pena"]
                    "examples": ["No.", "-", "A", "T", ".", "nada"]
                }
            }
        }
        }
    }
}


Aquí están los "VERBATIM" para clasificar:

{text}

Respuestas esperadas (solo etiquetas numéricas):
"""

#with open('C:\\Users\\dcanet\\Downloads\\septiembre\\septiembre.json', 'r', encoding="utf-8") as f:
#    classified_comments = json.load(f)

# Función para clasificar un comentario usando OpenAI API
def classify_comment(comment_text):
    prompt = f"""

    Instrucciones:
    1. Clasifica las siguientes "VERBATIM" según el plan de códigos.
    2. Devuelve solo las etiquetas numéricas correspondientes a cada "VERBATIM".
    3. No proporciones ninguna explicación o descripción, solo las etiquetas numéricas.
    4. Revisa y asegúrate de que cada etiqueta sea correcta antes de contestar.

    Dado el siguiente plan de codificación y sus definiciones:
    {json.dumps(prompt_template)}

    Ayúdate de las descripciones, las condiciones y los ejemplos proporcionados.

    Aquí están los "VERBATIM" para clasificar:
    {comment_text}

    Respuestas esperadas (solo etiquetas numéricas):
    """

    response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un asistente que clasifica comentarios."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=50,
    temperature=0
)

# Procesar la respuesta
    return response.choices[0].message['content'].strip()

# Aplicar la función a cada comentario del DataFrame
tqdm.pandas(desc="Clasificando comentarios")
df['pred'] = df['VERBATIM'].progress_apply(classify_comment)

# Guardar el DataFrame actualizado en un nuevo archivo
df.to_csv('C:\\Users\\dcanet\\Downloads\\septiembre\\resultados_10xCat.csv', index=False)