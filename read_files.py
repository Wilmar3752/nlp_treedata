import os
import re
from nltk.corpus import stopwords
import pandas as pd
import spacy
import es_core_news_md


def clean_tokenize(texto):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation). La funcion tambien elimina stop words y realiza una
    lematizacion de palablas en español
    '''
    # lista de stop words en español
    stop_words = stopwords.words('spanish')
    # Cargando nlp para lematizar
    nlp = es_core_news_md.load()
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    #nuevo_texto = nuevo_texto.split(sep = ' ')
    nuevo_texto = nlp(nuevo_texto)
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token.lemma_ for token in nuevo_texto if len(token) > 1 if not str(token) in stop_words]
    
    return(nuevo_texto)

def get_data(path) -> pd.DataFrame:
    dir_list = [i for i in os.listdir(path)]
    x = []
    for dir in dir_list:
        files_dir = [i for i in os.listdir(path+'/'+dir+'/')]
        for file in files_dir:
            with open(path+'/'+dir+'/'+file,"r") as f:
                texto = f.read()
                salida = [texto,dir,file]
                x.append(salida)
    df = pd.DataFrame(x,columns = ['document','topic','file'] )
    df['clean_text'] = df['document'].apply(clean_tokenize)
    return df

    

if __name__ == "__main__":
    print("Leyendo, limpiando y preparando data")
    data = get_data("data/")
    print("Escribiendo el .csv final")
    data.to_csv('data_training.csv',encoding="utf-8")
    print("Procesamiento finalizado con exito!")