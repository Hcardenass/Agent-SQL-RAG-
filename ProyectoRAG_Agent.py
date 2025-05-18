import os
import pandas as pd
import streamlit as st
from utils import view_name, get_schema, run_query, get_field_desc
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core. prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonREPLTool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import sys
import io
import textwrap
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.tools import Tool
import getpass
from langchain_google_community import GmailToolkit

import re
#pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# ———————————————————————————
# Lee tu API key de OpenAI
# ———————————————————————————
with open("./api_key.txt") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

#with open("./api_langchain.txt") as archivo:
#  apikeylg = archivo.read()

#os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
#os.environ["LANGCHAIN_API_KEY"] = apikeylg
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_PROJECT"] = "TecnoProject"


# 1) Inicialización de LLM y embeddings
def _init_llm_and_embeddings():
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return llm, embeddings

# 2) Definición de loaders originales

def load_web_tool():
    llm, embeddings = _init_llm_and_embeddings()
    hyde = HypotheticalDocumentEmbedder.from_llm(llm, embeddings, "web_search")
    urls_with_titles = [
        ("https://tecnofil.com.pe/nosotros/quienes-somos/", "Quienes Somos | Tecnofil"),
        ("https://tecnofil.com.pe/noticias/", "Noticias | Tecnofil"),
        ("https://tecnofil.com.pe/nosotros/procesos/", "Procesos | Tecnofil"),
        ("https://tecnofil.com.pe/nosotros/certificaciones/", "Certificaciones | Tecnofil"),
        ("https://tecnofil.com.pe/nosotros/nuestra-gente/", "Nuestra Gente | Tecnofil"),
        ("https://tecnofil.com.pe/productos/", "Productos | Tecnofil"),
    ]
    docs = []
    for url, title in urls_with_titles:
        loaded = WebBaseLoader(url).load()
        for doc in loaded:
            doc.metadata['title'] = title
        docs.extend(loaded)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)

    def web_query(query: str) -> str:
        m = re.search(r'"([^\"]+)"', query)
        title = m.group(1) if m else query
        hyde_emb = hyde.embed_query(title)
        results = vector_store.similarity_search_by_vector(hyde_emb, k=5)
        snippets = [f"- ({d.metadata.get('title')}) {d.page_content[:200].strip()}..." for d in results]
        return "\n".join(snippets)

    return Tool(
        name="empresa_tecnofil",
        func=web_query,
        description="Recupera información de la sección de Tecnofil basada en el título proporcionado."
    )


def load_email_tool():
    emails_folder = os.path.join(os.getcwd(), 'Emails')
    docs = []
    for fname in os.listdir(emails_folder):
        if fname.lower().endswith('.eml'):
            loader = UnstructuredEmailLoader(os.path.join(emails_folder, fname))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    _, embeddings = _init_llm_and_embeddings()
    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(chunks)
    retriever = vs.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    return retriever.as_tool(
        name='informacion_de_email',
        description='Extrae información de los emails almacenados en la carpeta ./Emails'
    )


def load_pdf_tool():
    pdf_dir = os.path.join(os.getcwd(), "ArchivosPDF")
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    _, embeddings = _init_llm_and_embeddings()
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever.as_tool(
        name="informacion_documentos_pdf",
        description="Extrae información de los documentos PDF almacenados en ./ArchivosPDF"
    )



# 3) Factory para carga lazy
def get_tool_lazy(loader_func):
    cached = None
    def wrapper(query: str) -> str:
        nonlocal cached
        if cached is None:
            cached = loader_func()
        return cached.func(query)
    tool = loader_func()
    return Tool(name=tool.name, func=wrapper, description=tool.description)

# 4) Definición de herramientas lazy
tool_web = get_tool_lazy(load_web_tool)
tool_email = get_tool_lazy(load_email_tool)
tool_pdf = get_tool_lazy(load_pdf_tool)

# 5) Configuración de modelo, prompts y cadenas SQL/Gráfico
model = ChatOpenAI(model="gpt-4o-mini", verbose=True)

# SQL Prompt
tprompt_sql = ChatPromptTemplate.from_template("""
Definición de la vista {view_name} y sus columnas :
{view_schema}

Significado de cada campo:
{field_desc}

**IMPORTANTE PARA GENERAR SQL EN SAP HANA**:  
- Usa comillas dobles (`\"`) para esquemas, tablas y columnas.  
- No utilices backticks ni corchetes.   
- Devuelve únicamente la sentencia SQL, sin bloques de código, sin punto y coma, ni texto adicional.

Pregunta: {question}

"""
)
sql_chain = (
    RunnablePassthrough.assign(view_schema=get_schema, view_name=lambda _: view_name, field_desc=get_field_desc)
    | tprompt_sql
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Prompt natural
tprompt_nl = ChatPromptTemplate.from_template("""
Vista consultada: {view_name}
Esquema de columnas:
{view_schema}

Pregunta: {question}
SQL generado: {query}
Resultados: {response}

**Guía de explicación**:  
- Para referirte a «primera posición» usa siempre el valor menor de `POSICION_PEDIDO`.  
- Devuelve los datos en formato tabular en respuesta a preguntas en lenguaje natural

Explica los resultados en lenguaje natural:
"""
)
full_chain = (
    RunnablePassthrough.assign(query=sql_chain)
      .assign(view_schema=get_schema, response=lambda vs: run_query(vs["query"]), view_name=lambda _: view_name)
    | tprompt_nl
    | model
)

# Exponer como herramienta gráfica
tool_grafico = PythonREPLTool(
    name="generar_grafico",
    description="Ejecuta un bloque de código Python para crear gráficos con pandas/matplotlib.",
    globals={"run_query": run_query}
)


# Exponer como herramienta SQL
tool_sql = full_chain.as_tool(
    name="busqueda_comercial",
    description=(
        "Genera y ejecuta consultas SQL sobre la vista Analytic_Model_Comercial, "
        "conociendo sus columnas y descripciones, "
        "y devuelve resultados en lista simple o tabulares en respuesta a preguntas en lenguaje natural."
    )
)

# 6) Construcción del agente
def build_agent():
    tolkit = [tool_sql, tool_grafico, tool_email, tool_pdf, tool_web]
    prompt = ChatPromptTemplate.from_messages([
        ('system', """
        Eres un asistente inteligente y gentil especializado en responder con herramientas.
        
        Herramientas disponibles:
        - busqueda_comercial: para generar y ejecutar consultas SQL sobre la vista `Analytic_Model_Comercial`.
        - generar_grafico: para crear cualquier tipo de gráfico (línea, barra, torta, dispersión, etc.) a partir de datos obtenidos.
        - informacion_de_email: para extraer información de correos electrónicos.
        - informacion_documentos_pdf: para extraer información de documentos PDF.
        - empresa_tecnofil: para recuperar información de la sección web de Tecnofil
        - enviar_email_gmail(destinatario: str, asunto: str, cuerpo: str, formato_tabular: bool=False)
Si el usuario solicita “envíame en formato tabular” o “mándalo en tabla”, el agente debe invocar enviar_email_gmail(...) con formato_tabular=True.
        Flujo de respuesta:
         **busqueda_comercial**: Usalo solo para generar y ejecutar todas las consultas SQL sobre la vista `Analytic_Model_Comercial`.
        - **generar_grafico**: solo si la pregunta implica análisis gráfico (p. ej. “evolución”, “tendencia”, “gráfica”):
             a. Ejecuta primero `busqueda_comercial` para obtener los datos.  
             b. Luego llama a generar_grafico con un bloque Python que:
                - importe pandas y matplotlib,
                - convierta los resultados en un DataFrame,
                - cree la carpeta "graficos" si no existe,
                - guarde el gráfico como un archivo PNG en "graficos/grafico.png" usando plt.savefig(),
                - NO use ni st.pyplot() ni plt.show()
             c. No muestres los datos en texto: solo las dos llamadas a herramienta en ese orden

        Instrucciones importantes:
        - Puedes invocar hasta dos herramientas, en este orden: SQL → gráfico.
        - Devuelve únicamente la salida cruda de la(s) herramienta(s).
        - Para SQL en SAP HANA usa comillas dobles y sin punto y coma.
        """),
        ('human', '{messages}')
    ])
    memory = MemorySaver()
    agent = create_react_agent(model, tools=tolkit, checkpointer=memory, prompt=prompt)
    return agent

# 7) Interfaz Streamlit
# 7) Interfaz Streamlit
def main():
    st.set_page_config(page_title="TecnoAI", page_icon="🤖")
    st.title("TecnoAI 🤖")
    st.markdown("**¡Bienvenido a TecnoAI! Haga su consulta.**")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = "user_001"
    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Escribe tu pregunta:")
    final_response = None
    if st.button("Ejecutar") and user_input:
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        for step in build_agent().stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
            final_response = step["messages"][-1].content
            step["messages"][-1].pretty_print()
        st.markdown(f"**Asistente:** {final_response}")
        st.session_state.history.extend([("Usuario", user_input), ("Asistente", final_response)])

    st.subheader("Historial de Conversación")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
    st.markdown(f"**Total pasos en historial:** {len(st.session_state.history)}")

if __name__ == "__main__":
    main()

##Cuantos pedidos tiene el cliente EATON FAYETTEVILLE en el mes de julio del 2025 y cuales son sin repetir?
#Necesito la lista completa de los lotes que contiene el email con nombre Schneider Monterrey Pickup - May 8, 2025
#detallame informacion del documento pdf de orden de compra del dia 09/05/2025
#Cuales son las ultimas noticias de tecnofil?
#Con que certificaciones cuenta Tecnofil?
#Detallame mas de sus certificaciones ISO
#hasme un resumen de los untos mas imporantes del documento PDF gestion de reportes BI

#pip install streamlit --upgrade
#Grafica un reporte en forma de torta del top 3 de cliente con más cantidad de colocado en el 2025