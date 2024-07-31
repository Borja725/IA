import pandas as pd
from tqdm import tqdm
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# Leer y preparar los datos
df = pd.read_csv("DATASETS\\Dataset_400_Mod_v3.csv", encoding="utf-8")
verbatims = df['VERBATIM'].to_numpy()
codes = df['CODES'].to_numpy()

# Cargar JSON de predicciones y recortar
dj = pd.read_json("JSON\\verbatim_predictions_llama3.1_dataset_Mejor_Version.json")
dj = dj.to_numpy()
dj = dj.tolist()



llm = ChatGroq(
    api_key="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",
    model="llama-3.1-70b-versatile",
    temperature=0
)

json_definition = """
{
  "Verbatims Negativos": {
    "Entregas": {
      "0101": {
        "descripcion": "El producto o la caja han llegado dañados. Esto puede incluir cualquier tipo de daño visible en el embalaje o en el producto mismo al momento de la recepción. Los daños pueden ser físicos, como rasgaduras, abolladuras o roturas.",
        "ejemplo": "El paquete llegó con la caja rota y el producto estaba dañado.",
        "conditions": ["dañado", "rota", "desgastado", "malo estado", "embalaje"],
        "details": "Este código se utiliza cuando el cliente reporta que el embalaje o el producto están en mal estado físico al momento de la entrega. Se centra en daños visibles que afectan la integridad del producto o su caja."
      },
      "0102": {
        "descripcion": "Comentarios sobre demoras en el tiempo de entrega del pedido que superan el plazo estimado. Este código se aplica cuando el retraso no se debe a un error en el pedido ni a problemas con la mensajería, sino a problemas logísticos generales.",
        "ejemplo": "Mi pedido estaba previsto para llegar el lunes, pero llegó el miércoles sin explicación.",
        "conditions": ["retraso", "demora", "plazo", "entrega tarde", "retraso entrega"],
        "details": "Este código abarca situaciones donde el pedido no llega dentro del plazo prometido o esperado sin una justificación adecuada. Incluye cualquier tipo de demora en la entrega que no está relacionada con problemas específicos del pedido o del servicio de mensajería."
      },
      "0103": {
        "descripcion": "Ha habido un error en el pedido, como recibir un producto incorrecto, productos adicionales no solicitados, o falta de productos. Se refiere a cualquier discrepancia en el contenido del pedido recibido en comparación con lo que fue solicitado.",
        "ejemplo": "Recibí un producto diferente al que pedí y faltaban algunos artículos.",
        "conditions": ["error", "producto equivocado", "faltante", "productos adicionales", "pedido incorrecto"],
        "details": "Este código cubre errores relacionados con la exactitud del pedido, tales como recibir productos incorrectos, faltantes, o adicionales. No incluye problemas de entrega, que se manejan con otros códigos. como el 101 para pedidos dañados"
      },
      "0104": {
        "descripcion": "Fallos debidos a la empresa de logística, como problemas con el servicio de entrega por parte de empresas de mensajería (por ejemplo, SEUR). Se refiere a problemas causados por la empresa de transporte, no por el proceso de pedido o el producto en sí.",
        "ejemplo": "El repartidor no entregó el paquete en mi dirección y tuve que ir a recogerlo a la oficina de mensajería.",
        "conditions": ["logística", "mensajería", "SEUR", "problemas entrega", "repartidor"],
        "details": "Este código se usa cuando los problemas de entrega son atribuibles a la empresa de logística en lugar de a Nespresso. Incluye problemas como retrasos en la entrega, entrega a la dirección incorrecta, o cualquier otra falla en el servicio de mensajería."
      }
    },
    "Mkt & Promociones": {
      "0201": {
        "descripcion": "Comentarios relacionados con el precio de los productos, ya sea queja por el precio, percepción de que es demasiado alto, o que ha cambiado. Incluye cualquier mención del costo que afecte la percepción del valor del producto.",
        "ejemplo": "El precio del producto ha subido mucho desde la última vez que lo compré.",
        "conditions": ["precio", "costoso", "aumento", "caro", "valor"],
        "details": "Este código cubre cualquier comentario que haga referencia al costo de los productos, ya sea que el cliente lo considere demasiado alto, haya notado un aumento en el precio, o tenga otras quejas relacionadas con el precio."
      },
      "0202": {
        "descripcion": "Comentarios sobre condiciones específicas como la necesidad de cambiar un producto, falta de promociones u ofertas, solicitudes de reembolsos, problemas con garantías(maquinas,etc...), quejas sobre gastos de envío excesivos, o reparaciones costosas. Incluye una amplia gama de problemas relacionados con las políticas comerciales de la empresa.",
        "ejemplo": "No encontré las ofertas que mencionaron en la publicidad y, además, los gastos de envío fueron muy altos.", "Todavía estoy esperando que cambien mi cafetera",
        "conditions": ["condiciones", "promociones", "ofertas", "gastos envío", "problemas con garantías", "reembolsos", "reparaciones"],
        "details": "Este código se usa para una variedad de problemas relacionados con la política comercial y las condiciones ofrecidas por Nespresso. Incluye cualquier queja sobre cambios en el producto, falta de ofertas, problemas con garantías, gastos de envío, o reparaciones."
      },
      "0203": {
        "descripcion": "Comentarios que expresan decepción o confusión sobre la publicidad de la empresa, especialmente si la publicidad se percibe como engañosa o no corresponde con la realidad del producto o servicio recibido. Esto incluye casos donde la publicidad generó expectativas que no se cumplieron.",
        "ejemplo": "La publicidad prometía una oferta especial, pero cuando compré el producto, no había descuento alguno.",
        "conditions": ["publicidad", "engañoso", "expectativas", "promesas", "anuncio"],
        "details": "Este código se aplica a situaciones donde la publicidad o promoción de Nespresso se percibe como engañosa o no cumple con lo prometido. Incluye cualquier discrepancia entre lo anunciado y la experiencia real del cliente."
      },
      "0204": {
        "descripcion": "Comentarios sobre el programa de fidelidad, incluyendo la necesidad de premiar mejor a los clientes frecuentes o mejorar el trato a los clientes leales.",
        "ejemplo": "Me gustaría que Nespresso ofreciera más beneficios para clientes frecuentes como yo.", "No valoran a los clientes",
        "conditions": ["fidelidad", "beneficios", "premios", "trato", "clientes frecuentes", "valorar"],
        "details": "Este código se utiliza para comentarios sobre cómo Nespresso podría mejorar el programa de fidelidad o el trato hacia los clientes leales. Incluye sugerencias para más beneficios o mejor reconocimiento para clientes frecuentes."
      },
      "0205": {
        "descripcion": "Comentarios sobre el plan de suscripción de Nespresso, incluyendo su funcionamiento, precios, opciones disponibles, o servicios incluidos en el plan.",
        "ejemplo": "El plan de suscripción tiene un buen precio, pero me gustaría más variedad en las opciones de café.",
        "conditions": ["suscripción", "plan", "precio", "opciones", "servicios"],
        "details": "Este código cubre comentarios específicos sobre el plan de suscripción de Nespresso, como aspectos relacionados con el precio, la variedad de opciones, o cualquier problema relacionado con el servicio de suscripción."
      },
      "0206": {
        "descripcion": "Comentarios sobre el café en sí, como la falta de variedades, escasez de café descafeinado, o problemas con la calidad general del café ofrecido.",
        "ejemplo": "No hay suficientes opciones de café descafeinado disponibles en la tienda.",
        "conditions": ["café", "variedades", "descafeinado", "calidad", "stock"],
        "details": "Este código se usa para problemas relacionados directamente con el café ofrecido por Nespresso, tales como falta de variedad, problemas de calidad, o disponibilidad de café descafeinado."
      },
      "0207": {
        "descripcion": "Comentarios sobre las máquinas de café, como problemas recurrentes con el servicio técnico, fallos en el funcionamiento, o mala calidad en general de las máquinas.",
        "ejemplo": "Mi cafetera ha pasado más tiempo en reparación que en uso. La calidad es bastante baja.",
        "conditions": ["máquina", "cafetera", "servicio técnico", "calidad", "problemas"],
        "details": "Este código cubre quejas sobre las máquinas de café, incluyendo problemas técnicos recurrentes, defectos de fabricación, o cualquier otro problema relacionado con la calidad y el funcionamiento de las máquinas."
      },
      "0208": {
        "descripcion": "Comentarios sobre productos que no encajan en las categorías anteriores. Pueden ser quejas o sugerencias sobre cualquier aspecto del producto que no se pueda clasificar bajo los códigos específicos mencionados.",
        "ejemplo": "El diseño de la cápsula es incómodo para usar.",
        "conditions": ["producto", "aspectos", "diseño", "usabilidad"],
        "details": "Este código se usa para comentarios sobre aspectos generales del producto que no se ajustan a las categorías previamente definidas. Incluye observaciones sobre el diseño, uso, o cualquier otro aspecto no especificado."
      }
    },
    "Reciclaje": {
      "0301": {
        "descripcion": "Comentarios sobre la falta de recogida de cápsulas usadas o la necesidad de proporcionar bolsas para reciclaje. Incluye la preocupación de que las cápsulas usadas no se recojan adecuadamente o que no se facilite el reciclaje.",
        "ejemplo": "No recogen las cápsulas usadas en mi área. Necesito que proporcionen bolsas para reciclaje.",
        "conditions": ["reciclaje", "cápsulas usadas", "bolsas", "recogida", "problemas"],
        "details": "Este código abarca problemas relacionados con la logística de reciclaje, como la falta de recogida de cápsulas usadas o la necesidad de recibir bolsas adecuadas para el reciclaje."
      },
      "0302": {
        "descripcion": "Comentarios sobre la sostenibilidad de las prácticas de la empresa y su impacto ambiental. Incluye preocupaciones sobre cómo la empresa maneja el impacto ambiental y las iniciativas ecológicas.",
        "ejemplo": "Aprecio que Nespresso esté tomando medidas para ser más ecológico y reducir su impacto ambiental.",
        "conditions": ["sostenibilidad", "impacto ambiental", "iniciativas ecológicas", "prácticas verdes"],
        "details": "Este código se usa para comentarios positivos sobre las iniciativas de sostenibilidad y las prácticas ecológicas de Nespresso. Incluye apreciaciones sobre los esfuerzos por reducir el impacto ambiental."
      }
    },
    "Servicio Técnico": {
      "0401": {
        "descripcion": "El producto sigue sin funcionar después de haber sido reparado por el servicio técnico. Incluye situaciones en las que el problema persiste a pesar de la intervención del servicio técnico.",
        "ejemplo": "Después de que repararon mi cafetera, sigue sin funcionar correctamente.",
        "conditions": ["servicio técnico", "reparación", "problema persistente", "fallo", "funcionamiento"],
        "details": "Este código se aplica cuando un producto sigue presentando problemas después de haber sido reparado. Incluye casos en los que la reparación no solucionó el problema inicial."
      },
      "0402": {
        "descripcion": "Tiempo de recogida mayor al estipulado para la recogida del producto por parte del servicio técnico. Incluye retrasos en la recolección del producto para su reparación.",
        "ejemplo": "El servicio técnico tardó más de una semana en recoger mi máquina para reparación, a pesar de que me dijeron que lo harían en 48 horas.",
        "conditions": ["recogida", "tiempo de espera", "retraso", "servicio técnico", "problemas"],
        "details": "Este código se usa para reportar retrasos en el proceso de recogida del producto por el servicio técnico. Incluye demoras que superan el tiempo prometido para la recogida."
      },
      "0403": {
        "descripcion": "Tiempo de entrega de la máquina reparada que es mayor al estipulado. Incluye retrasos en la devolución del producto después de la reparación.",
        "ejemplo": "Mi máquina reparada se entregó dos semanas más tarde de lo que se había prometido.",
        "conditions": ["entrega", "máquina reparada", "retraso", "tiempo de entrega", "servicio técnico"],
        "details": "Este código se utiliza cuando hay demoras en la entrega de un producto después de haber sido reparado. Incluye cualquier retraso en la devolución del producto al cliente."
      },
      "0404": {
        "descripcion": "Lentitud en la comunicación con el servicio técnico, incluyendo problemas para obtener actualizaciones sobre el estado de la reparación o la falta de seguimiento adecuado.",
        "ejemplo": "No pude obtener actualizaciones sobre el estado de mi reparación durante varias semanas.",
        "conditions": ["comunicación", "servicio técnico", "lenta", "seguimiento", "actualización"],
        "details": "Este código cubre problemas relacionados con la comunicación y el seguimiento del servicio técnico. Incluye lentitud en proporcionar información o actualizaciones sobre el estado del servicio."
      },
      "0405": {
        "descripcion": "Problemas con la máquina de sustitución proporcionada durante la reparación. Esto incluye situaciones donde la máquina de reemplazo tiene defectos o no cumple con las expectativas del cliente.",
        "ejemplo": "La máquina de sustitución que me dieron estaba defectuosa y no funcionaba bien.",
        "conditions": ["máquina sustitución", "defectuosa", "calidad", "reemplazo"],
        "details": "Este código se usa para reportar problemas con la máquina de sustitución ofrecida durante el período de reparación. Incluye fallos o deficiencias en la máquina proporcionada."
      },
      "0406": {
        "descripcion": "Comentarios sobre la calidad del servicio técnico en general. Incluye quejas sobre la eficacia, profesionalidad, o cualquier aspecto relacionado con la calidad del servicio técnico recibido.",
        "ejemplo": "El servicio técnico fue muy ineficaz y no resolvieron mi problema adecuadamente.",
        "conditions": ["servicio técnico", "calidad", "ineficaz", "profesionalidad", "queja"],
        "details": "Este código se aplica a comentarios que critican la calidad general del servicio técnico. Incluye quejas sobre la falta de eficacia, profesionalidad"
      },
      "0495": {
        "descripcion": "Comentarios sobre el servicio técnico que no se pueden clasificar en ninguno de los códigos anteriores. Incluye cualquier otro problema relacionado con el servicio técnico que no encaja en las categorías establecidas.",
        "ejemplo": "Tuve un problema con la reparación, pero no encaja en ninguna categoría específica.",
        "conditions": ["servicio técnico", "problema general", "no clasificado", "caso único"],
        "details": "Este código cubre problemas relacionados con el servicio técnico que no se ajustan a las categorías específicas previamente definidas. Incluye casos que no encajan claramente en otros códigos."
      }
    },
    "Website": {
      "0501": {
        "descripcion": "Problemas con la página web o el sistema de pago online. Incluye cualquier dificultad para completar una transacción, errores en la página, o problemas técnicos relacionados con el proceso de compra en línea.",
        "ejemplo": "Intenté hacer una compra en la web, pero el sistema de pago no aceptó mi tarjeta y no podía completar la transacción.",
        "conditions": ["página web", "pago online", "errores técnicos", "funcionalidad", "transacción"],
        "details": "Este código se utiliza para reportar problemas específicos relacionados con la funcionalidad del sitio web o el proceso de pago online. Incluye errores técnicos que impiden completar la compra."
      }
    },
    "Trade": {
      "0601": {
        "descripcion": "Mala atención al cliente por parte de terceros, como tiendas asociadas o plataformas de venta. Incluye problemas relacionados con la atención recibida de empresas que venden los productos de Nespresso, como Amazon o El Corte Inglés.",
        "ejemplo": "La atención al cliente en El Corte Inglés fue muy deficiente y no resolvieron mi problema.",
        "conditions": ["atención", "terceros", "Amazon", "El Corte Inglés", "deficiente"],
        "details": "Este código se aplica a comentarios sobre la atención al cliente proporcionada por terceros que venden productos Nespresso, y no directamente por Nespresso."
      }
    },
    "BTQ": {
      "0701": {
        "descripcion": "Mala atención al cliente en las tiendas físicas de Nespresso. Incluye cualquier queja sobre el trato recibido en las tiendas físicas, como falta de amabilidad, servicio lento, o asistencia inadecuada.",
        "ejemplo": "El personal en la tienda física de Nespresso fue grosero y no me ayudó con mi problema.",
        "conditions": ["atención", "tienda física", "Nespresso", "personal", "grosero"],
        "details": "Este código cubre quejas sobre la atención al cliente en las tiendas físicas de Nespresso, incluyendo problemas con la actitud o comportamiento del personal."
      },
      "0702": {
        "descripcion": "Problemas con el proceso de degustación del producto en la tienda física. Incluye quejas sobre la organización, calidad o la experiencia general durante las sesiones de degustación en las tiendas.",
        "ejemplo": "La experiencia de degustación en la tienda fue desorganizada y poco informativa.",
        "conditions": ["degustación", "tienda física", "desorganizado", "proceso", "informativo"],
        "details": "Este código se utiliza para reportar problemas relacionados con el proceso de degustación del producto en las tiendas físicas, como una mala organización o falta de información durante la degustación."
      },
      "0708": {
        "descripcion": "Falta de tiendas físicas de Nespresso, cierres o falta de presencia en ciertas ubicaciones. Incluye comentarios sobre la insuficiencia de tiendas en áreas específicas o el cierre de tiendas existentes.",
        "ejemplo": "No hay suficientes tiendas de Nespresso en mi área, y las que hay están muy lejos.",
        "conditions": ["falta de tiendas", "ubicaciones", "presencia", "cierres", "distancia"],
        "details": "Este código cubre la falta de tiendas físicas de Nespresso, problemas relacionados con el número de tiendas, cierres de tiendas, o la ubicación geográfica de las tiendas existentes."
      }
    },
    "CRC": {
      "0808": {
        "descripcion": "Comentarios sobre la atención al cliente a través del servicio de atención telefónica, incluyendo falta de soluciones, personal ineficaz, y mala gestión de incidencias. Este código cubre casos donde la atención recibida no fue satisfactoria o no se resolvió el problema del cliente.",
        "ejemplo": "El representante del CRC no resolvió mi problema y fue poco útil en su asistencia.",
        "conditions": ["atención", "teléfono", "soluciones", "ineficaz", "gestión de incidencias"],
        "details": "Este código se aplica a problemas con el servicio de atención al cliente recibido a través del teléfono, incluyendo la falta de soluciones adecuadas, mal manejo de las incidencias, o cualquier otra deficiencia en el servicio telefónico."
      }
    },
    "Otros Negativos": {
      "0095": {
        "descripcion": "Otros problemas negativos que no se han codificado anteriormente. Incluye cualquier queja o comentario negativo que no encaja en las categorías previamente establecidas.",
        "ejemplo": "Tuve un problema con mi cuenta que no se encaja en ninguna de las categorías anteriores.",
        "conditions": ["problemas", "no clasificados", "general", "no específico"],
        "details": "Este código se usa para problemas generales que no se ajustan a las categorías específicas. Incluye cualquier otra queja negativa que no encaje claramente en los códigos existentes."
      }
    },
    "Verbatims Positivos": {
      "Entregas": {
        "9101": {
          "descripcion": "Satisfacción general con la entrega del producto. Incluye comentarios positivos sobre la puntualidad y el estado en que el producto llegó.",
          "ejemplo": "El paquete llegó a tiempo y en perfectas condiciones.",
          "conditions": ["entrega", "puntualidad", "estado", "satisfacción"],
          "details": "Este código cubre comentarios positivos relacionados con la entrega del producto, incluyendo la puntualidad y el estado en que el producto llegó al cliente."
        }
      },
      "Mkt & Promociones": {
        "9201": {
          "descripcion": "Satisfacción con el producto recibido. Incluye comentarios positivos sobre la calidad, características, o cualquier otro aspecto positivo del producto.",
          "ejemplo": "El café tiene un excelente sabor y es justo lo que buscaba.",
          "conditions": ["producto", "calidad", "satisfacción", "características"],
          "details": "Este código se aplica a comentarios positivos sobre el producto en sí, abarcando aspectos como la calidad, el sabor, y la satisfacción general con el producto."
        },
        "9202": {
          "descripcion": "Satisfacción con la estrategia de publicidad de la marca. Incluye comentarios positivos sobre cómo se presentan los productos y las promociones, y la percepción general de la publicidad.",
          "ejemplo": "La publicidad de Nespresso es clara y atractiva, y realmente refleja la calidad del producto.",
          "conditions": ["publicidad", "estrategia", "promociones", "claridad"],
          "details": "Este código se usa para comentarios positivos sobre la forma en que la marca presenta sus productos y promociones. Incluye apreciaciones sobre la claridad y efectividad de la publicidad."
        },
        "9203": {
          "descripcion": "Satisfacción con los beneficios proporcionados por el programa de fidelidad de Nespresso. Incluye comentarios positivos sobre cómo se reconocen y premian a los clientes leales.",
          "ejemplo": "Estoy muy contento con los beneficios que recibo por ser un cliente frecuente de Nespresso.",
          "conditions": ["fidelidad", "beneficios", "premios", "clientes frecuentes"],
          "details": "Este código cubre comentarios positivos relacionados con el programa de fidelidad, incluyendo la satisfacción con los premios y beneficios ofrecidos a los clientes frecuentes."
        }
      },
      "Reciclaje": {
        "9301": {
          "descripcion": "Satisfacción con las prácticas de sostenibilidad y reciclaje de la empresa. Incluye comentarios positivos sobre las iniciativas ecológicas y la gestión de reciclaje.",
          "ejemplo": "Aprecio que Nespresso esté tomando medidas para mejorar la sostenibilidad y el reciclaje.",
          "conditions": ["sostenibilidad", "reciclaje", "prácticas ecológicas", "impacto ambiental"],
          "details": "Este código se aplica a comentarios positivos sobre las iniciativas de sostenibilidad y reciclaje de Nespresso. Incluye apreciaciones sobre los esfuerzos por reducir el impacto ambiental y fomentar el reciclaje."
        }
      },
      "Servicio Técnico": {
        "9401": {
          "descripcion": "Satisfacción general con el servicio técnico recibido. Incluye comentarios positivos sobre la eficacia, rapidez y calidad del servicio de reparación.",
          "ejemplo": "El servicio técnico fue excelente, resolvieron mi problema rápidamente y de manera profesional.",
          "conditions": ["servicio técnico", "calidad", "rapidez", "eficaz", "satisfacción"],
          "details": "Este código cubre comentarios positivos sobre la experiencia general con el servicio técnico, incluyendo la eficacia, profesionalismo y rapidez en la resolución de problemas."
        }
      },
      "Website": {
        "9501": {
          "descripcion": "Satisfacción con la página web en general. Incluye comentarios positivos sobre la funcionalidad, diseño y experiencia de usuario del sitio web.",
          "ejemplo": "La página web de Nespresso es fácil de navegar y muy intuitiva.",
          "conditions": ["página web", "funcionalidad", "diseño", "experiencia de usuario"],
          "details": "Este código se usa para comentarios positivos sobre la página web de Nespresso, abarcando aspectos como el diseño, la funcionalidad, y la facilidad de navegación."
        }
      },
      "Trade": {
        "9601": {
          "descripcion": "Satisfacción con la atención al cliente proporcionada por terceros, como tiendas asociadas. Incluye comentarios positivos sobre la atención recibida de empresas que venden productos Nespresso.",
          "ejemplo": "La atención al cliente en Amazon fue excelente y ayudaron rápidamente con mi consulta.",
          "conditions": ["atención", "terceros", "Amazon", "El Corte Inglés", "satisfacción"],
          "details": "Este código se aplica a comentarios positivos sobre la atención al cliente de terceros que venden productos Nespresso, como Amazon o El Corte Inglés."
        }
      },
      "BTQ": {
        "9701": {
          "descripcion": "Satisfacción con la atención al cliente en las tiendas físicas de Nespresso. Incluye comentarios positivos sobre el trato recibido en las tiendas físicas, como la amabilidad del personal y la calidad del servicio.",
          "ejemplo": "El personal en la tienda de Nespresso fue muy amable y servicial.",
          "conditions": ["atención", "tienda física", "Nespresso", "personal", "amabilidad"],
          "details": "Este código cubre comentarios positivos sobre la atención al cliente en las tiendas físicas de Nespresso, incluyendo aspectos como la amabilidad y eficacia del personal."
        },
        "9702": {
          "descripcion": "Satisfacción con el proceso de degustación del producto en la tienda física. Incluye comentarios positivos sobre la organización y la experiencia durante las sesiones de degustación en las tiendas.",
          "ejemplo": "La degustación en la tienda fue muy bien organizada y me ayudó a elegir el café perfecto.",
          "conditions": ["degustación", "tienda física", "organización", "experiencia", "satisfacción"],
          "details": "Este código se usa para comentarios positivos sobre el proceso de degustación en las tiendas físicas, incluyendo la calidad de la experiencia y la organización del evento."
        }
      },
      "CRC": {
        "9801": {
          "descripcion": "Satisfacción con la atención recibida a través del servicio de atención telefónica, incluyendo la profesionalidad del personal y la eficacia en la resolución de problemas.",
          "ejemplo": "El servicio al cliente por teléfono fue muy profesional y resolvió mi problema de manera eficiente.",
          "conditions": ["atención", "teléfono", "profesionalidad", "resolución", "satisfacción"],
          "details": "Este código se aplica a comentarios positivos sobre la atención al cliente recibida a través del teléfono, abarcando la profesionalidad del personal y la eficacia en resolver problemas."
        }
      },
      "Otros Positivos": {
        "9995": {
          "descripcion": "Otros comentarios positivos que no se han codificado anteriormente. Incluye cualquier apreciación o comentario positivo que no encaja en las categorías previamente establecidas.",
          "ejemplo": "Tuve una experiencia muy positiva con Nespresso, pero no encaja en ninguna categoría específica.",
          "conditions": ["comentarios positivos", "no clasificados", "general", "aprecio"],
          "details": "Este código se usa para comentarios positivos que no se ajustan a las categorías específicas. Incluye cualquier otra apreciación que no encaje claramente en los códigos existentes."
        }
      }
    },
    "NS / NC": {
      "0009": {
        "descripcion": "Comentarios donde el cliente no sabe o no contesta. Incluye casos en los que la información proporcionada es insuficiente o el cliente no ofrece una respuesta clara.",
        "ejemplo": "No tengo una opinión sobre esto.",
        "conditions": ["no sabe", "no contesta", "información insuficiente", "respuesta vaga"],
        "details": "Este código se utiliza para casos en los que el cliente no proporciona una respuesta clara o no sabe cómo responder a la pregunta. Incluye respuestas vagas o falta de información."
      }
    }
  }
}

"""

# Plantilla de prompt
prompt_template = """
Tarea de Clasificación:

1. Objetivo: Asigna una etiqueta numérica a cada fragmento de texto marcado como "VERBATIM", utilizando el "contexto" y las "unidades" proporcionadas. Asegúrate de que la etiqueta refleje adecuadamente el sentimiento general del comentario y el aspecto específico abordado (por ejemplo, entrega, servicio técnico, promociones, etc.).

2. Formato de Respuesta: Solo etiquetas numéricas. No incluyas texto adicional, explicaciones ni descripciones.

3. Referencia:

  3.1. Utiliza la "semantic_conclusion" para guiar la clasificación en términos de sentimiento.
  
  3.2. Consulta las "details" del JSON para comparar con los "description" del plan de códigos, y asegúrate de que el comentario se ajuste a las condiciones de las categorías de codificación.
  
  3.3 Usa la "location" del JSON para identificar si el comentario es sobre "a distancia" o en "tienda".

  3.4 Usa el "channel_of_communication" del JSON para determinar si es "telefónico" o "presencial".

  3.5 Considera el "sentiment" para clasificar el comentario en la categoría correspondiente.

  3.6 Verifica los códigos que tienen "Example" en el plan de códigos, ya que estos pueden ser más específicos y ayudarte a evitar confusiones.

4. Estrategia de Clasificación:

  4.1 Comentarios negativos: Identifica si el comentario se relaciona con problemas en la entrega, el servicio técnico, las promociones, el reciclaje, el sitio web, la atención de terceros, o la atención en tienda. Asigna la etiqueta negativa correspondiente según el contexto.
  
  4.2 Comentarios positivos: Identifica si el comentario se relaciona con la satisfacción en la entrega, la calidad de los productos, la estrategia publicitaria, el reciclaje, el servicio técnico, el sitio web, la atención de terceros, o la atención en tienda. Asigna la etiqueta positiva adecuada.

5. Criterios Específicos:

  5.1 Entregas: Usa etiquetas de la categoría de entregas si el comentario menciona problemas o satisfacciones con la entrega de productos.

  5.2 Promociones: Usa etiquetas de la categoría de marketing y promociones si el comentario menciona promociones, ofertas, o publicidad.

  5.3 Reciclaje: Usa etiquetas de reciclaje si el comentario menciona la gestión de cápsulas usadas o preocupaciones ambientales.

  5.4 Servicio Técnico: Usa etiquetas de servicio técnico si el comentario menciona problemas con la reparación o la recogida de productos.

  5.5 Sitio Web: Usa etiquetas de la categoría de sitio web si el comentario menciona problemas con la página web o el pago en línea.

  5.6 Terceros: Usa etiquetas de atención al cliente por parte de terceros si el comentario menciona experiencias con minoristas o terceros.

  5.7 Tienda: Usa etiquetas de atención en tienda si el comentario menciona la experiencia en una tienda física de Nespresso.

  5.8 Precisión y Consistencia: Asegúrate de revisar cuidadosamente las condiciones y ejemplos en el plan de códigos para evitar malentendidos y errores en la clasificación.

Definición del JSON y Plan de Codificación:

```json
{json_definition}
```
Fragmentos de Texto para Clasificar:

{text}

Respuesta Esperada:

(Solo etiquetas numéricas. No incluyas texto adicional ni explicaciones.)
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "json_definition"])
chain = LLMChain(llm=llm, prompt=prompt)

# Función para clasificar el texto
def assign_tags(text):
    result = chain.invoke({"text": text, "json_definition": json_definition})
    return result['text']

# Clasificar los datos
results = []
for verb, item, code in tqdm(zip(verbatims, dj, codes), total=len(verbatims)):
    tags = assign_tags(item)
    results.append({"VERBATIM": verb, "CODES": code, "PREDICTIONS": tags})

# Guardar resultados en CSV
pd.DataFrame(results).to_csv('verbatim_predictions_llama3-70b_v1_prompt_engin.csv', index=False)