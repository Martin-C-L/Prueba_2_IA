import streamlit as st
import os
import wikipedia
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from fpdf import FPDF  

st.set_page_config(page_title="Investigador AI", page_icon="🕵️‍♂️", layout="wide")

def create_pdf(text):
    """Convierte el texto markdown a un PDF simple"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    

    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, txt=text)
    
    return pdf.output(dest='S').encode('latin-1')

def setup_api():
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://models.inference.ai.azure.com")
    
    if not api_key:
        with st.sidebar:
            st.warning("⚠️ No se detectó API KEY en el entorno.")
            api_key = st.text_input("Ingresa tu GitHub/Azure Token:", type="password")
            api_base = st.text_input("Endpoint Base:", value=api_base)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
        return True
    return False


class WikipediaSearchTool(BaseTool):
    name: str = "BuscadorWikipedia"
    description: str = "Busca en Wikipedia un tema."
    
    def _run(self, query: str) -> str:
        try:
            wikipedia.set_lang("es")
            return wikipedia.summary(query, sentences=5)
        except Exception as e:
            return f"Error buscando: {str(e)}"

def ejecutar_investigacion(tema):
    wiki_tool = WikipediaSearchTool()
    
    investigador = Agent(
        role="Investigador Senior",
        goal=f"Investigar a fondo sobre: {tema}",
        backstory="Experto en encontrar datos precisos y fuentes confiables.",
        tools=[wiki_tool],
        verbose=True,
        allow_delegation=False
    )

    redactor = Agent(
        role="Redactor Técnico",
        goal="Escribir un artículo atractivo y bien estructurado.",
        backstory="Escritor especializado en sintetizar información compleja.",
        verbose=True,
        allow_delegation=False
    )

    tarea1 = Task(
        description=f"Busca información detallada sobre '{tema}'. Encuentra historia, datos clave y curiosidades.",
        expected_output="Lista de puntos clave y datos verificados.",
        agent=investigador
    )

    tarea2 = Task(
        description="Usando la investigación, escribe un post de blog en formato texto claro (evita markdown complejo) con títulos y viñetas.",
        expected_output="Artículo completo en texto plano.",
        agent=redactor,
        context=[tarea1]
    )

    crew = Crew(
        agents=[investigador, redactor],
        tasks=[tarea1, tarea2],
        process=Process.sequential
    )

    return crew.kickoff()


st.title("🕵️‍♂️ Agente Investigador (PDF)")
st.markdown("---")

api_lista = setup_api()

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=150)
    st.info("**Genera reportes y descárgalos en PDF.**")

with col2:
    tema = st.text_input("¿Sobre qué quieres investigar hoy?", placeholder="Ej: Inteligencia Artificial...")
    boton_buscar = st.button("🚀 Iniciar Investigación", type="primary", disabled=not api_lista)

if boton_buscar and tema:
    with st.spinner('🤖 Investigando y generando PDF...'):
        try:
            resultado_crew = ejecutar_investigacion(tema)
            texto_resultado = str(resultado_crew) 
            
            st.success("¡Investigación completada!")
            
            st.markdown("### 📄 Vista Previa")
            st.markdown(texto_resultado)
            st.markdown("---")

            if hasattr(resultado_crew, 'token_usage'):
                usage = resultado_crew.token_usage
                st.subheader("📊 Estadísticas")
                k1, k2, k3 = st.columns(3)
                k1.metric("Total Tokens", usage.total_tokens)
                k2.metric("Input", usage.prompt_tokens)
                k3.metric("Output", usage.completion_tokens)

            pdf_bytes = create_pdf(texto_resultado)

            st.download_button(
                label="📕 Descargar Informe en PDF",
                data=pdf_bytes,
                file_name=f"informe_{tema}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

elif boton_buscar and not tema:
    st.warning("Por favor escribe un tema primero.")