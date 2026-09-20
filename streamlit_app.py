import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Nombre, descripción y color de cada ODS, tal como los publica la ONU.
ODS_INFO = {
    1: {
        "nombre": "Fin de la pobreza",
        "descripcion": "Poner fin a la pobreza en todas sus formas en todo el mundo.",
        "color": "#E5243B",
        "icono": "🏠"
    },
    2: {
        "nombre": "Hambre cero",
        "descripcion": "Poner fin al hambre, lograr la seguridad alimentaria y la mejora de la nutrición y promover la agricultura sostenible.",
        "color": "#DDA63A",
        "icono": "🌾"
    },
    3: {
        "nombre": "Salud y bienestar",
        "descripcion": "Garantizar una vida sana y promover el bienestar para todos en todas las edades.",
        "color": "#4C9F38",
        "icono": "🏥"
    },
    4: {
        "nombre": "Educación de calidad",
        "descripcion": "Garantizar una educación inclusiva, equitativa y de calidad y promover oportunidades de aprendizaje durante toda la vida.",
        "color": "#C5192D",
        "icono": "📚"
    },
    5: {
        "nombre": "Igualdad de género",
        "descripcion": "Lograr la igualdad entre los géneros y empoderar a todas las mujeres y las niñas.",
        "color": "#FF3A21",
        "icono": "⚖️"
    },
    6: {
        "nombre": "Agua limpia y saneamiento",
        "descripcion": "Garantizar la disponibilidad de agua y su gestión sostenible y el saneamiento para todos.",
        "color": "#26BDE2",
        "icono": "💧"
    },
    7: {
        "nombre": "Energía asequible y no contaminante",
        "descripcion": "Garantizar el acceso a una energía asequible, segura, sostenible y moderna.",
        "color": "#FCC30B",
        "icono": "⚡"
    },
    8: {
        "nombre": "Trabajo decente y crecimiento económico",
        "descripcion": "Promover el crecimiento económico inclusivo y sostenible, el empleo y el trabajo decente para todos.",
        "color": "#A21942",
        "icono": "💼"
    },
    9: {
        "nombre": "Industria, innovación e infraestructura",
        "descripcion": "Construir infraestructuras resilientes, promover la industrialización inclusiva y sostenible y fomentar la innovación.",
        "color": "#FD6925",
        "icono": "🏭"
    },
    10: {
        "nombre": "Reducción de las desigualdades",
        "descripcion": "Reducir la desigualdad en y entre los países.",
        "color": "#DD1367",
        "icono": "🤝"
    },
    11: {
        "nombre": "Ciudades y comunidades sostenibles",
        "descripcion": "Lograr que las ciudades y los asentamientos humanos sean inclusivos, seguros, resilientes y sostenibles.",
        "color": "#FD9D24",
        "icono": "🏙️"
    },
    12: {
        "nombre": "Producción y consumo responsables",
        "descripcion": "Garantizar modalidades de consumo y producción sostenibles.",
        "color": "#BF8B2E",
        "icono": "🔄"
    },
    13: {
        "nombre": "Acción por el clima",
        "descripcion": "Adoptar medidas urgentes para combatir el cambio climático y sus efectos.",
        "color": "#3F7E44",
        "icono": "🌍"
    },
    14: {
        "nombre": "Vida submarina",
        "descripcion": "Conservar y utilizar sosteniblemente los océanos, los mares y los recursos marinos para el desarrollo sostenible.",
        "color": "#0A97D9",
        "icono": "🐟"
    },
    15: {
        "nombre": "Vida de ecosistemas terrestres",
        "descripcion": "Proteger, restablecer y promover el uso sostenible de los ecosistemas terrestres y detener la pérdida de biodiversidad.",
        "color": "#56C02B",
        "icono": "🌳"
    },
    16: {
        "nombre": "Paz, justicia e instituciones sólidas",
        "descripcion": "Promover sociedades pacíficas e inclusivas para el desarrollo sostenible y facilitar el acceso a la justicia para todos.",
        "color": "#00689D",
        "icono": "🕊️"
    },
    17: {
        "nombre": "Alianzas para lograr los objetivos",
        "descripcion": "Fortalecer los medios de ejecución y revitalizar la Alianza Mundial para el Desarrollo Sostenible.",
        "color": "#19486A",
        "icono": "🌐"
    }
}

# Carpetas donde puede estar el modelo según dónde corra la app
# (repo, Streamlit Cloud o el Drive de Colab donde se entrenó).
DIRECTORIOS_MODELOS = [
    '.',
    'modelos',
    'modelos_guardados',
    'resources/models',
    '/content/drive/MyDrive/Colab Notebooks/Micro proyecto 2/modelos_guardados',
]

PIPELINES_E2E = ['pipeline_ods_e2e.joblib', 'pipeline_ods_lsa.joblib']

# El voting pesa más de 100 MB y no está en GitHub, por eso va primero
# solo para cuando se corre en local.
CLASIFICADORES = ['voting_clf.joblib', 'xgboost_clf.joblib', 'random_forest_clf.joblib']

DATOS_ENTRENAMIENTO = [
    'Office Open XML spreadsheet.xlsx',
    'Train_textosODS.xlsx',
    '../Proyecto 2/Office Open XML spreadsheet.xlsx',
    '../Proyecto 2/Train_textosODS.xlsx',
]


def limpiar_texto_input(texto: str) -> str:
    """Pasa a minúsculas y deja solo letras y espacios (sin URLs ni correos)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'\S+@\S+', ' ', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def _primer_existente(rutas):
    return next((r for r in rutas if os.path.exists(r)), None)


def _cargar_pipeline_e2e():
    ruta = _primer_existente(
        os.path.join(d, nombre) for nombre in PIPELINES_E2E for d in DIRECTORIOS_MODELOS
    )
    if ruta is None:
        return None

    obj = joblib.load(ruta)
    # Algunas versiones se guardaron como dict junto con el label encoder.
    if isinstance(obj, dict) and 'pipeline' in obj:
        return obj['pipeline'], obj.get('label_encoder')
    return obj, None


def _cargar_por_componentes():
    for carpeta in DIRECTORIOS_MODELOS:
        ruta_extractor = os.path.join(carpeta, 'best_feature_extractor.joblib')
        ruta_clf = _primer_existente(os.path.join(carpeta, c) for c in CLASIFICADORES)
        if not os.path.exists(ruta_extractor) or ruta_clf is None:
            continue

        ruta_le = os.path.join(carpeta, 'label_encoder.joblib')
        le = joblib.load(ruta_le) if os.path.exists(ruta_le) else None
        pipe = Pipeline([
            ('extractor', joblib.load(ruta_extractor)),
            ('clf', joblib.load(ruta_clf)),
        ])
        return pipe, le
    return None


def _entrenar_respaldo():
    ruta = _primer_existente(DATOS_ENTRENAMIENTO)
    if ruta is None:
        return None

    df = pd.read_excel(ruta)
    textos = df['textos'].apply(limpiar_texto_input)
    le = LabelEncoder()
    y = le.fit_transform(df['ODS'])

    pipe = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),
        ('tfidf', TfidfTransformer()),
        ('tsvd', TruncatedSVD(n_components=100, random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(textos, y)
    return pipe, le


@st.cache_resource
def cargar_modelo():
    # Primero el pipeline completo, después las piezas sueltas y, si no hay
    # nada guardado, se entrena un random forest sencillo con el Excel.
    for cargar in (_cargar_pipeline_e2e, _cargar_por_componentes, _entrenar_respaldo):
        resultado = cargar()
        if resultado is not None:
            return resultado
    return None, None


st.set_page_config(
    page_title="Clasificador de ODS",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .ods-card {
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-top: 16px;
        margin-bottom: 24px;
    }
    .ods-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .ods-desc {
        font-size: 15px;
        line-height: 1.5;
        opacity: 0.95;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Sobre el proyecto")
    st.write(
        "Esta app clasifica textos en español según los 17 Objetivos de Desarrollo Sostenible (ODS) de la ONU."
    )
    st.markdown("---")
    st.markdown("**Cómo funciona el modelo**")
    st.write(
        "Limpiamos el texto con expresiones regulares y lematización, lo vectorizamos con BoW y TF-IDF "
        "y lo reducimos a 100 componentes con TruncatedSVD (LSA). Sobre ese espacio clasifica un XGBoost "
        "o un ensamble por votación."
    )
    st.markdown("---")
    st.caption("Microproyecto 2 - ML No Supervisado")

st.title("Clasificador de textos ODS")
st.write(
    "Pegue un párrafo, por ejemplo de una política pública o de un informe técnico, "
    "y el modelo le dice con cuál ODS se relaciona."
)

pipeline, label_encoder = cargar_modelo()

if pipeline is None:
    st.error("No hay un modelo guardado ni datos de entrenamiento para armar uno. Revise la carpeta modelos/.")
    st.stop()


def a_numero_ods(clase) -> int:
    if label_encoder is not None:
        clase = label_encoder.inverse_transform([clase])[0]
    return int(clase)


ejemplos = {
    "Elija un ejemplo...": "",
    "Formación de docentes (ODS 4)": "Los maestros de preescolar y jardín de infancia deben ser licenciados universitarios con formación pedagógica continua y evaluación nacional.",
    "Agua y cuencas (ODS 6)": "El acceso a fuentes de agua potable y saneamiento básico en comunidades rurales depende de la gestión sostenible de las cuencas y el tratamiento de aguas residuales.",
    "Brecha salarial de género (ODS 5)": "Se requiere cerrar la brecha salarial entre hombres y mujeres e incrementar la participación femenina en puestos de liderazgo directivo.",
    "Energía renovable (ODS 7)": "La transición hacia matrices de generación solar y eólica permite reducir la dependencia de combustibles fósiles y garantizar energía limpia y asequible.",
    "Justicia y transparencia (ODS 16)": "El fortalecimiento del estado de derecho y la transparencia judicial son indispensables para combatir la corrupción y garantizar la paz social."
}

seleccion = st.selectbox("Textos de prueba:", list(ejemplos.keys()))

texto_usuario = st.text_area(
    "Texto a analizar:",
    value=ejemplos[seleccion],
    height=150,
    placeholder="Escriba o pegue aquí el texto en español..."
)

if st.button("Clasificar", type="primary"):
    if not texto_usuario.strip():
        st.warning("Escriba un texto antes de clasificar.")
    else:
        texto_limpio = limpiar_texto_input(texto_usuario)

        prediccion_ods = a_numero_ods(pipeline.predict([texto_limpio])[0])

        info = ODS_INFO.get(prediccion_ods, {
            "nombre": f"ODS {prediccion_ods}",
            "descripcion": "Objetivo de Desarrollo Sostenible",
            "color": "#333333",
            "icono": "🎯"
        })

        st.subheader("Resultado")

        st.markdown(f"""
        <div class="ods-card" style="background-color: {info['color']};">
            <div class="ods-title">{info['icono']} ODS {prediccion_ods}: {info['nombre']}</div>
            <div class="ods-desc">{info['descripcion']}</div>
        </div>
        """, unsafe_allow_html=True)

        # No todos los clasificadores que probamos exponen predict_proba.
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([texto_limpio])[0]
            top3 = np.argsort(probs)[::-1][:3]

            st.markdown("#### Los tres ODS más probables")
            for puesto, idx in enumerate(top3, start=1):
                ods = a_numero_ods(pipeline.classes_[idx])
                nombre = ODS_INFO.get(ods, {}).get("nombre", f"ODS {ods}")
                st.write(f"**{puesto}. ODS {ods} ({nombre})**: {probs[idx] * 100:.1f} %")
                st.progress(float(probs[idx]))
