from langchain_community.llms import Ollama
from tqdm import tqdm
#llm = Ollama(model="llama3")  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `

#print(llm("Dime un chiste, máximo 50 palabras"))
import pandas as pd
from spacy_llm.util import assemble

df = pd.read_csv("verbatims_nov_2023_Version_17072024.csv")
df = df.head(3)
# Cargar el modelo
results=[]
threshold=0.8

batch_size = 50

# Cargar el modelo
nlp = assemble("config.cfg")

verb=df["VERBATIM"].to_numpy()

codes = df["CODES"].to_numpy()

# Iterating over the data with tqdm for progress visualization
for verbatim, code, doc in tqdm(zip(verb, codes, nlp.pipe(verb, batch_size=50)), total=len(verb)):
    predic = [label for label, prob in doc.cats.items() if prob > threshold]
    results.append({"VERBATIM": verbatim, "CODES": code, "PREDICTIONS": predic})


# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)

# Guardar en un archivo CSV
df_results.to_csv('verbatim_predictions_few_shotV4.csv', index=False)

