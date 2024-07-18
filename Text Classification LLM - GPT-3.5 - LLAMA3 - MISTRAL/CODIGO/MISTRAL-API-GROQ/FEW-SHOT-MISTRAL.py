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

Ahora te voy a pasar ejemplos de comentarios clasificados, con text que sera la frase, abajo te pondre otro text que sera la palabra clave, en label te pondre la etiqueta que corresponde a ese texto clave y hace que la frase de arriba tenga esa etiqueta y abajo te pondre answer que sera a que etiqueta corresponde la frase al completo.

    "text": "A pesar de la calidad del café, la falta de cuidado en el embalaje resulta en cápsulas dañadas con frecuencia.",
    "spans": 
      
        "text": "cápsulas dañadas",
        "is_entity": True,
        "label": "0101"
      
    
     "answer": "0101"
  
  
    "text": "El servicio de entrega es pésimo, cada vez peor. El último pedido tardó 11 días para recibirlo.",
    "spans": 
      
        "text": "pedido tardó 11 días",
        "is_entity": True,
        "label": "0102"
      
     "answer": "0102"
  
  
    "text": "El pedido llegó incompleto, y no puedo disfrutar de mi café como lo planeé.",
    "spans": 
      
        "text": "pedido llegó incompleto",
        "is_entity": True,
        "label": "0103"
      
     "answer": "0103"
  
  
    "text": "La empresa de mensajería no ha sido confiable en la entrega de mis pedidos de Nespresso.",
    "spans": 
      
        "text": "empresa de mensajería no ha sido confiable",
        "is_entity": True,
        "label": "0104"
      
     "answer": "0104"
  
  
    "text": "El repartidor de MRW entregó mi paquete tarde y no agradeceron ninguna explicación ni disculpa.",
    "spans": 
      
        "text": "repartidor",
        "is_entity": True,
        "label": "0104"
      
      
        "text": "de MRW",
        "is_entity": True,
        "label": "0104"
      
      
        "text": "entregó mi paquete tarde",
        "is_entity": True,
        "label": "0102"
      
     "answer": "0104,0102"
  
  
    "text": "Aunque el cafe es bueno, Nespresso resulta bastante caro.",
    "spans": 
       
        "text": "Aunque el cafe es bueno",
        "is_entity": True,
        "label": "9201"
      
      
        "text": "bastante caro",
        "is_entity": True,
        "label": "0201"
      
     "answer": "9201,0201"
  
  
    "text": "Encuentro que el café es particularmente delicioso y bien preparado. No puedo evitar sentir que me están estafando con estos precios.",
    "spans": 
       
        "text": "café es particularmente delicioso",
        "is_entity": True,
        "label": "9201"
      
      
        "text": "No puedo evitar sentir que me están estafando con estos precios",
        "is_entity": True,
        "label": "0201"
      
     "answer": "9201,0201"
  
  
    "text": "No puedo evitar sentir que me están estafando con estos precios.",
    "spans": 
      
        "text": "estafando con estos precios",
        "is_entity": True,
        "label": "0201"
      
     "answer": "0201"
  
  
    "text": "Compré una cafetera que tuvo problemas repetidos, y cuando ya no estaba en garantía, me decepcionaron.",
    "spans": 
      
        "text": "garantia, me decepcionaron",
        "is_entity": True,
        "label": "0202"
      
     "answer": "0202"
  
  
    "text": "Compré un producto basado en su publicidad, pero no era lo que prometían. Decepcionante.",
    "spans": 
      
        "text": "publicidad, pero no era lo que prometían",
        "is_entity": True,
        "label": "0203"
      
     "answer": "0203"
  
  
    "text": "No han hecho lo que prometieron.",
    "spans": 
      
        "text": "No han hecho lo que prometieron",
        "is_entity": True,
        "label": "0203"
      
     "answer": "0203"
  
  
    "text": "Como cliente de muchos años, esperaba un trato especial, pero las ofertas no reflejan mi fidelidad.",
    "spans": 
      
        "text": "esperaba un trato especial",
        "is_entity": True,
        "label": "0204"
      
      
        "text": "las ofertas no reflejan mi fidelidad",
        "is_entity": True,
        "label": "0204"
      
     "answer": "0204"
  
  
    "text": "El plan de suscripción es un desastre. No cumplen con las entregas.",
    "spans": 
      
        "text": "plan de suscripcion es un desastre",
        "is_entity": True,
        "label": "0205"
      
     "answer": "0205"
  
  
    "text": "Creo que la calidad de las cápsulas, es decir del café, ha cambiado, ya no es la que era. Es de inferior calidad que hace unos años.",
    "spans": 
      
        "text": "inferior calidad",
        "is_entity": True,
        "label": "0206"
      
     "answer": "0206"
  
  
    "text": "Creo que falta variedad en términos de tés y bebidas saludables en el catálogo de Nespresso.",
    "spans": 
      
        "text": "falta variedad",
        "is_entity": True,
        "label": "0206"
      
     "answer": "0206"
  
  
    "text": "Mi máquina no hace café correctamente y el depósito de agua pierde.",
    "spans": 
      
        "text": "Mi máquina no hace café correctamente",
        "is_entity": True,
        "label": "0207"
      
     "answer": "0207"

    "text": "La máquina da problemas constantemente Pedí un reembolso y la empresa nunca devolvió el dinero. Es una falta de respeto al cliente.",
    "spans": 
      
        "text": "La máquina da problemas constantemente",
        "is_entity": True,
        "label": "0207"
      
        "text": "Pedí un reembolso y la empresa nunca devolvió el dinero",
        "is_entity": True,
        "label": "0202"
      
        "text": "Es una falta de respeto al cliente",
        "is_entity": True,
        "label": "0202"
      
     "answer": "0207,0202"
  
    "text": "Siempre tengo que llamar al servicio técnico. Mi cafetera está en reparación y prometieron proporcionarme una máquina de cortesía, nunca llegó.",
    "spans": 
        "text": "Mi cafetera está en reparación",
        "is_entity": True,
        "label": "0207"
      
        "text": "prometieron proporcionarme una máquina de cortesía, nunca llegó",
        "is_entity": True,
        "label": "0405"
      
     "answer": "0207,0405"
  
    "text": "Disfruto mucho del producto, aunque me gustaría una mayor facilidad de compra.",
    "spans": 
      
        "text": "me gustaria una mayor facilidad de compra",
        "is_entity": True,
        "label": "0208"
      
     "answer": "0208"
  
    "text": "Nada que mencionar.",
    "spans": 
      
        "text": "Nada que mencionar",
        "is_entity": True,
        "label": "0009"
      
     "answer": "0009"
 
    "text": ".",
    "spans": 
      
        "text": ".",
        "is_entity": True,
        "label": "0009"
      
     "answer": "0009"
  
    "text": "Tiene margen de mejora. No fue la mejor experiencia.",
    "spans": 
      
        "text": "margen de mejora",
        "is_entity": True,
        "label": "0095"
      
      
        "text": "No fue la mejor experiencia",
        "is_entity": True,
        "label": "0095"
      
     "answer": "0095"
  
  
    "text": "Error tras error",
    "spans": 
      
        "text": "Error tras error",
        "is_entity": True,
        "label": "0095"
      
     "answer": "0095"
  
    "text": "Se debería dar solución al problema de las cápsulas difíciles de reciclar.",
    "spans": 
      
        "text": "cápsulas difíciles de reciclar",
        "is_entity": True,
        "label": "0301"
      
     "answer": "0301"
  
    "text": "Debería haber más opciones ecológicas para las cápsulas de café en el mercado.",
    "spans": 
      
        "text": "haber más opciones ecológicas",
        "is_entity": True,
        "label": "0302"
      
     "answer": "0302"

    "text": "La calidad del producto es insuperable.",
    "spans": 
      
        "text": "calidad del producto es insuperable",
        "is_entity": True,
        "label": "9201"
      
     "answer": "9201"
  
    "text": "La calidad del café es inigualable.",
    "spans": 
      
        "text": "calidad del café es inigualable",
        "is_entity": True,
        "label": "9201"
      
     "answer": "9201"
  
    "text": "Mi máquina de café aún tiene problemas de funcionamiento después de ser reparada.",
    "spans": 
      
        "text": "después de ser reparada",
        "is_entity": True,
        "label": "0401"
      
        "text": "aún tiene problemas de funcionamiento",
        "is_entity": True,
        "label": "0401"
      
     "answer": "0401"
  
    "text": "No puedo comprender la tardanza en la recogida de mi máquina para reparación, es realmente frustrante.",
    "spans": 
      
        "text": " tardanza en la recogida",
        "is_entity": True,
        "label": "0402"
      
     "answer": "0402"
  
  
    "text": "Estoy decepcionado con el tiempo que el servicio técnico ha tardado en arreglar mi cafetera.",
    "spans": 
      
        "text": "ha tardado en arreglar",
        "is_entity": True,
        "label": "0403"
      
     "answer": "0403"
  
    "text": "La lentitud de comunicación con el soporte técnico ha dificultado poder disfrutar de mi cafetera sin preocupaciones.",
    "spans": 
      
        "text": "lentitud de comunicación",
        "is_entity": True,
        "label": "0404"
      
     "answer": "0404"
  
    "text": "La máquina de reemplazo tenía un botón de encendido que no funcionaba correctamente.",
    "spans": 
      
        "text": "no funcionaba correctamente",
        "is_entity": True,
        "label": "0405"
      
      
        "text": "máquina de reemplazo",
        "is_entity": True,
        "label": "0405"
      
     "answer": "0405"
  
  
    "text": "La falta de profesionalidad en el servicio técnico es una constante decepción.",
    "spans": 
      
        "text": "servicio técnico es una constante decepción",
        "is_entity": True,
        "label": "0406"
      
     "answer": "0406"
  
    "text": "La lentitud de su sitio web y aplicaciones es inaceptable en estos tiempos.",
    "spans": 
      
        "text": "lentitud de su sitio web y aplicaciones",
        "is_entity": True,
        "label": "0501"
      
     "answer": "0501"
  
  
    "text": "Compré una cafetera en El Corte Inglés y ahora me enfrento a problemas constantes con ella.",
    "spans": 
      
        "text": "cafetera en El Corte Inglés y ahora me enfrento a problemas constantes",
        "is_entity": True,
        "label": "0601"
      
     "answer": "0601"
  
  
    "text": "El personal de la tienda debería mejorar su actitud.",
    "spans": 
      
        "text": "personal de la tienda",
        "is_entity": True,
        "label": "0701"
      
      
        "text": "debería mejorar su actitud",
        "is_entity": True,
        "label": "0701"
      
     "answer": "0701"
  
  
    "text": "Desearía poder probar el café antes de decidirme a comprarlo.",
    "spans": 
      
        "text": "poder probar el café",
        "is_entity": True,
        "label": "0702"
      
     "answer": "0702"
  
    "text": "El cierre de la tienda en el centro de Granada fue una decisión poco acertada.",
    "spans": 
      
        "text": "cierre de la tienda",
        "is_entity": True,
        "label": "0708"
      
      
        "text": "fue una decisión poco acertada",
        "is_entity": True,
        "label": "0708"
      
     "answer": "0708"
  
    "text": "Mi última comunicación fue con el servicio técnico para la revisión de mi cafetera, pero no he recibido respuesta a mis correos electrónicos.",
    "spans": 
      
        "text": "servicio tecnico",
        "is_entity": True,
        "label": "0808"
      
      
        "text": "no he recibido respuesta a mis correos electrónicos",
        "is_entity": True,
        "label": "0808"
      
     "answer": "0808"
  
    "text": "La entrega de pedidos es excelente  Buen servicio",
    "spans": 
      
        "text": "entrega de pedidos es excelente",
        "is_entity": True,
        "label": "9101"
      
     "answer": "9101"
  
    "text": "Un producto de calidad con una amplia variedad de opciones.",
    "spans": 
      
        "text": "producto de calidad",
        "is_entity": True,
        "label": "9201"
      
      
        "text": "amplia variedad de opciones",
        "is_entity": True,
        "label": "9201"
      
     "answer": "9201"
  
    "text": "La marca ha revolucionado el concepto de marketing en su sector.",
    "spans": 
      
        "text": "ha revolucionado el concepto de marketing",
        "is_entity": True,
        "label": "9202"
      
     "answer": "9202"
  
  
    "text": "Como miembro del club, disfruto de recompensas que no encuentro en otros lugares.",
    "spans": 
      
        "text": "disfruto de recompensas",
        "is_entity": True,
        "label": "9203"
      
     "answer": "9203"
  
  
    "text": "Confío en una marca que valora y promueve la sostenibilidad.",
    "spans": 
      
        "text": "valora y promueve la sostenibilidad",
        "is_entity": True,
        "label": "9301"
      
     "answer": "9301"
  
  
    "text": "Buen servicio de reparación y máquina de repuesto.",
    "spans": 
      
        "text": "Buen servicio de reparación",
        "is_entity": True,
        "label": "9401"
      
     "answer": "9401"
  
  
    "text": "La web es clara y fácil de usar, y siempre llega rápido.",
    "spans": 
      
        "text": "La web es clara y fácil de usar",
        "is_entity": True,
        "label": "9501"
      
     "answer": "9501"
  
  
    "text": "Los dependientes de Amazon son atentos y cumplen con las solicitudes de los clientes de manera efectiva.",
    "spans": 
      
        "text": "dependientes de Amazon son atentos",
        "is_entity": True,
        "label": "9601"
      ,
      
        "text": "cumplen con las solicitudes de los clientes de manera efectiva",
        "is_entity": True,
        "label": "9601"
      
     "answer": "9601"
  
  
    "text": "Los empleados de Nespresso siempre están dispuestos a ayudar.",
    "spans": 
      
        "text": "empleados de Nespresso",
        "is_entity": True,
        "label": "9701"
      
      
        "text": "dispuestos a ayudar",
        "is_entity": True,
        "label": "9701"
      
     "answer": "9701"
 
  
    "text": "Durante la degustación, puedo explorar las diferentes variedades de café.",
    "spans": 
      
        "text": "Durante la degustación",
        "is_entity": True,
        "label": "9702"
      
      
        "text": "puedo explorar las diferentes variedades",
        "is_entity": True,
        "label": "9702"
      
     "answer": "9702"
  
  
    "text": "La atención personal muy buena y los cafés excelentes",
    "spans": 
      
        "text": "atención personal muy buena",
        "is_entity": True,
        "label": "9801"
      
      
        "text": "cafes excelentes",
        "is_entity": True,
        "label": "9201"
      
     "answer": "9801,9201"
  
  
    "text": "El servicio es rápido y eficiente, y el trato siempre es amable y profesional. Disfruto del rico sabor y la variedad de opciones que ofrece.",
    "spans": 
      
        "text": "El servicio es rápido y eficiente",
        "is_entity": True,
        "label": "9801"
      
      
        "text": "trato siempre es amable y profesional",
        "is_entity": True,
        "label": "9801"
    
      
        "text": "Disfruto del rico sabor",
        "is_entity": True,
        "label": "9201"
      
     "answer": "9801,9201"
  
  
    "text": "La conducta amable y la destreza profesional del personal de la empresa.",
    "spans": 
      
        "text": "conducta amable",
        "is_entity": True,
        "label": "9801"
      
      
        "text": "destreza profesional del personal",
        "is_entity": True,
        "label": "9801"
      
     "answer": "9801"
  
  
    "text": "Todo genial.",
    "spans": 
      
        "text": "Todo genial",
        "is_entity": True,
        "label": "9995"

    "answer": "9995"
  
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