#!pip install google-cloud-storage
#pip install --upgrade --quiet  twilio
import os
import streamlit as st
from utils import view_name, get_schema, run_query, get_field_desc
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core. prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonREPLTool

from langchain.tools import tool

from openai import OpenAI

from google.cloud import storage
import os
from typing import Optional
import uuid
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_core.vectorstores import InMemoryVectorStore
import re
# ———————————————————————————
# 1) Lee tu API key de OpenAI
# ———————————————————————————
with open("./api_key.txt") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), "gc-tecno-dw.json")

# 1. Función para generar habla (speech) a partir de texto usando GPT-4o-mini-tts

client = OpenAI()
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=30)
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

def generate_speech_from_text(text: str, file_name: Optional[str] = None, folder_path: Optional[str] = "audios", voice: str = "alloy") -> str:
    """
    Genera un archivo de audio .mp3 a partir de un texto dado y lo guarda localmente.
    Devuelve la ruta local del archivo generado.
    """
    os.makedirs(folder_path, exist_ok=True)

    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_name = f"speech_{timestamp}_{unique_id}.mp3"

    output_path = os.path.join(folder_path, file_name)

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

@tool
def generar_speech(text: str) -> str:
    """
    Convierte texto en audio, guarda el archivo .mp3 localmente,
    lo sube automáticamente a GCS y muestra la URL pública en la terminal.
    """
    local_path = generate_speech_from_text(text)
    bucket_name = "gc-tecno-"
    public_url = upload_mp3_to_gcs(local_path, bucket_name)

    # Mostrar URL solo en terminal
    print(f"\n📢 URL pública del audio generado: {public_url}\n")

    # Retornar solo path local o mensaje para mantener limpio el output del bot
    return "El audio ha sido generado y guardado correctamente."


def upload_mp3_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: Optional[str] = None) -> str:
    """
    Sube un archivo .mp3 a un bucket de Google Cloud Storage en la carpeta 'audios/' y lo hace público.
    Retorna la URL pública del archivo.
    """
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"El archivo '{local_file_path}' no existe.")

    if destination_blob_name is None:
        # Subirá a una carpeta virtual "audios/" en el bucket
        destination_blob_name = f"audios/{os.path.basename(local_file_path)}"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path)
    blob.make_public()

    return blob.public_url


@tool
def upload_mp3_file_to_cloud_storage(local_path: str) -> str:
    """
    Sube un archivo de audio (.mp3) al bucket público de Google Cloud Storage y devuelve la URL pública.
    Bucket: gc-tecno-dw-pyt-001_cloudbuild
    """
    bucket_name = "gc-tecno-dw-pyt-001_cloudbuild"
    return upload_mp3_to_gcs(local_path, bucket_name)


from langchain_community.utilities.twilio import TwilioAPIWrapper

# Configura Twilio
twilio = TwilioAPIWrapper(
    account_sid="###",
    auth_token="###",
    from_number="whatsapp:+141",  # Este es el número de sandbox de WhatsApp (generalmente)
)

@tool
def send_audio_message_to_whatsapp(to_number: str, audio_url: str) -> str:
    """
    Envía un mensaje de audio (voz) a WhatsApp usando Twilio y una URL pública (audio_url debe ser .mp3 o .ogg).
    """
    from twilio.rest import Client

    client = Client(twilio.account_sid, twilio.auth_token)

    message = client.messages.create(
        body="🎧 Aquí tienes tu audio generado:",
        from_=twilio.from_number,
        to=f"whatsapp:{to_number}",
        media_url=[audio_url]
    )

    return f"✅ Mensaje enviado a {to_number}. SID: {message.sid}"


@tool
def generar_y_enviar_audio(texto: str, numero_destino: str) -> str:
    """
    Genera un audio desde texto, lo sube a GCS y lo envía por WhatsApp usando Twilio.
    """
    # 1. Generar audio localmente
    local_path = generate_speech_from_text(texto)

    # 2. Subir a GCS y obtener URL pública
    bucket_name = "gc-tecno-dw-pyt-001_cloudbuild"
    public_url = upload_mp3_to_gcs(local_path, bucket_name)

    # 3. Enviar vía WhatsApp
    resultado_envio = send_audio_message_to_whatsapp.invoke({
        "to_number": numero_destino,
        "audio_url": public_url
    })


    # 4. Retornar resultado
    return f"{resultado_envio}\n🔗 URL del audio: {public_url}"

# ———————————————————————————
# 4) Configurar modelo y prompts
model = ChatOpenAI(model="gpt-4o-mini",verbose=True) #gpt-4.1

promptsql = ChatPromptTemplate.from_template("""
Definición de la vista {view_name} y sus columnas :
{view_schema}

Significado de cada campo:
{field_desc}

**IMPORTANTE PARA GENERAR SQL EN SAP HANA**:  
- Usa comillas dobles (`"`) para esquemas, tablas y columnas.  
- No utilices backticks ni corchetes.   
- Devuelve únicamente la sentencia SQL, sin bloques de código, sin punto y coma, ni texto adicional.

Pregunta: {question}


""")

sql_chain = (
    RunnablePassthrough.assign(view_schema=get_schema, view_name=lambda _: view_name, field_desc = get_field_desc)
    | promptsql
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

promptnl = ChatPromptTemplate.from_template("""
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
""")


full_chain = (
    RunnablePassthrough.assign(query=sql_chain)
      .assign(view_schema=get_schema, response=lambda vs: run_query(vs["query"]), view_name=lambda _: view_name)
    | promptnl
    | model
)
# ———————————————————————————
# 7) Exponer como herramienta
# ———————————————————————————

tool_grafico = PythonREPLTool(
    name="generar_grafico",
    description="Ejecuta un bloque de código Python para crear gráficos con pandas/matplotlib.",
    # Aquí inyectamos run_query
    globals={"run_query": run_query}
)

tool_sql = full_chain.as_tool(
    name="busqueda_comercial",
    description=(
       "Genera y ejecuta consultas SQL sobre la vista Analytic_Model_Comercial, "
       "conociendo sus columnas  y descripciones, "
       "y devuelve resultados tabulares en respuesta a preguntas en lenguaje natural."
                )
)

memory = MemorySaver()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        Eres un asistente inteligente y gentil especializado en responder con herramientas.

    
          Herramientas disponibles:
            - busqueda_comercial: para generar y ejecutar consultas SQL sobre la vista `Analytic_Model_Comercial`.
            - generar_grafico: para crear cualquier tipo de gráfico (línea, barra, torta, dispersión, etc.) a partir de datos obtenidos.
            - informacion_de_email: para extraer información de correos electrónicos.
            - informacion_documentos_pdf: para extraer información de documentos PDF.
            - empresa_tecnofil: para recuperar información de la sección web de Tecnofil  
                    
          Flujo de respuesta:
            - **busqueda_comercial**: Usalo solo para generar y ejecutar todas las consultas SQL sobre la vista `Analytic_Model_Comercial`.  
            - **generar_grafico**: solo si la pregunta implica generar algun grafico o reporte (p. ej. “evolución”, “tendencia”, “gráfica”):
                 a. Ejecuta primero `busqueda_comercial` para obtener los datos.  
                 b. Luego llama a generar_grafico con un bloque Python que:
                    - importe pandas y matplotlib,
                    - convierta los resultados en un DataFrame,
                    - cree la carpeta "graficos" si no existe,
                    - guarde el gráfico como un archivo PNG en "graficos/grafico.png" usando plt.savefig(),
                    - NO use ni st.pyplot() ni plt.show()
                 c. No muestres los datos en texto: solo las dos llamadas a herramienta en ese orden
                 d. Cuando termines de usar la herramienta generar_grafico indica que se ha generado el grafico correctamente y que lo revise
                 e. No vuelves a generar otra vez el gráfico si es que no lo solicita.
                 
            - **generar_y_enviar_audio**: úsala si el usuario dice algo como “envíalo por WhatsApp”, “mándamelo al número...”, “quiero el resumen en audio por WhatsApp”, etc.
                 a. El texto debe generarse antes por el modelo (usa el resultado anterior o haz un resumen si aplica).
                 b. Llama a `generar_y_enviar_audio` pasándole el texto y el número de WhatsApp.
                 c. Internamente esta herramienta:
                    - genera el audio con voz,
                    - lo sube a GCS,
                    - y lo envía por WhatsApp.
                 d. No muestres la URL ni los pasos internos al usuario.
                 e. Solo confirma que el audio fue enviado con éxito.   
            
          Instrucciones importantes:
            - Puedes invocar hasta dos herramientas, en este orden: SQL → gráfico.
            - Devuelve únicamente la salida cruda de la(s) herramienta(s).
            - Para SQL en SAP HANA usa comillas dobles y sin punto y coma.
        """),

        ("human", "{messages}"),
    ]
)

tolkit = [tool_sql, tool_grafico, generar_speech, generar_y_enviar_audio, tool_web, tool_email, tool_pdf]
agent = create_react_agent(model, tools=tolkit, checkpointer=memory, prompt=prompt)

def build_message_history(history):
    messages = []
    for role, msg in history:
        if role == "Usuario":
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))
    return messages


# -------------------------------------
# 2. Interfaz Streamlit (al final)

import streamlit as st
from langchain.schema import HumanMessage

st.set_page_config(page_title="TecnoAI", page_icon="🤖", layout="wide")

# Inyectar CSS para fondo blanco y padding
st.markdown(
    """
    <style>
    /* Fondo blanco y texto negro por defecto */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stApp * {
        color: #000000 !important;
    }
    /* Centrar el contenedor principal */
    .block-container {
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Centrar títulos y subtítulos */
    .css-18e3th9 {  /* Selector para el título principal */
        text-align: center;
    }
    .css-1d391kg {  /* Selector para subtítulos o markdown */
        text-align: center;
    }
    /* Botón Ejecutar: fondo azul oscuro y texto blanco */
    .stButton>button {
        background-color: #0073e6;
        color: #FFFFFF !important;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    /* Texto y fondo del input de texto: fondo negro, texto blanco, texto alineado a la izquierda */
    .stTextInput>div>div>input {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        text-align: left;
    }
    /* Placeholder del input en gris claro */
    .stTextInput>div>div>input::placeholder {
        color: #DDDDDD !important;
    }
    /* Etiqueta del input centrada */
    label[for^="escribe-tu-pregunta"] {
        color: #000000 !important;
        text-align: center;
        width: 100%;
        display: block;
    }
    /* Alerta de éxito (respuesta del asistente): fondo verdoso y texto negro, centrada */
    .stAlert-success {
        background-color: #d4edda !important;
        color: #000000 !important;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicializar estado
if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = "user_001"

with st.sidebar:
    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.experimental_rerun()

st.title("TecnoAI 🤖")
st.markdown("**¡¡Bienvenido a TecnoAI! Haga su consulta.**")

user_input = st.text_input("Escribe tu pregunta:")

final_response = None
if st.button("Ejecutar") and user_input:
    # Construir historial temporal con el nuevo mensaje
    temp_history = st.session_state.history.copy()
    temp_history.append(("Usuario", user_input))

    # Convertir historial a mensajes para el agente
    messages = build_message_history(temp_history)

    config = {"configurable": {"thread_id": st.session_state.session_id}}

    with st.spinner("Generando la respuesta, por favor espere…"):
        for step in agent.stream(
            {"messages": messages},
            config,
            stream_mode="values",
        ):
            final_response = step["messages"][-1].content
            step["messages"][-1].pretty_print()

    st.success(f"Asistente: {final_response}")

    # Guardar en historial
    st.session_state.history.append(("Usuario", user_input))
    st.session_state.history.append(("Asistente", final_response))

# Mostrar historial de conversación
st.subheader("Historial de Conversación")
for role, msg in st.session_state.history:
    if role == "Usuario":
        st.markdown(f"**{role}:** {msg}")
    else:
        st.success(f"**{role}:** {msg}")

st.markdown(f"**Total de mensajes en historial:** {len(st.session_state.history)}")


#¿Cuáles son 3 primeros pedidos de venta?
# dame informacion del pedido 100613
# generame una grafica de barras de esos 3 pedidos
#ahora del pedido 101618
#Quien es el ejecutivo comercial del pedido 100613
#Grafica un reporte en forma de torta del top 3 de cliente con más cantidad de colocado en el 2025
#¿cuando es la fecha de ingreso del pedido 105103?
#¿Cuales son los 4 pedidos más reciente que ingreso el ejecutivo comercial luis ricardo hidalgo?
#¿Cuál es el pedido más reciente que ingreso el ejecutivo comercial luis ricardo hidalgo? RESPONDIDO
#Cuantos pedidos tiene el cliente EATON FAYETTEVILLE en el mes de julio del 2025 y cuales son?
#Indicame el top 3 de clientes con más cantidad de colocado en el 2025
#"Genial, envíame tu resultado anterior al correo adriancardenasc19@gmail.com  con el asunto 'Top Clientes 2025'"
#enviame tu resultado al numero whatsApp +51902705234