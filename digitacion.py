import os
import time
from google import genai
from PIL import Image
import cv2
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# === Carpetas ===
input_folder = "imagenes"
output_excel = "resultados.xlsx"


def texto_a_dataframe(texto):
    """
    Convierte el texto a un DataFrame.
    """
    texto = texto.strip()
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    csv_data = "\n".join(lineas)
    try:
        df = pd.read_csv(StringIO(csv_data), sep="|")
        df.columns = [c.strip().lower() for c in df.columns]
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df
    except Exception as e:
        print("⚠️ Error al parsear la tabla:", e)
        return pd.DataFrame()

def extraer_textos(imagenes):
    """
    Extrae el texto de las imágenes.
    """

    prompt = (
        """Recibirás imágenes de formularios de digitación escritos a mano con esta estructura:
        - Nombre y apellidos
        - Cédula
        - Edad
        - Teléfono
        - (El género no aparece explícito, pero debe inferirse a partir del nombre)
        - Direccion

        Instrucciones:
        - Extrae los datos de cada persona de forma precisa.
        - Devuelve ÚNICAMENTE una tabla sin texto adicional ni explicaciones.
        - La tabla debe tener exactamente estas columnas:
        nombre | apellidos | cedula | edad | telefono | genero | direccion
        - Usa un formato claro de tabla o CSV, ideal para convertir directamente en Excel.

        Detalles importantes:
        - El teléfono debe tener 7 o 10 dígitos.
        - Si un dato no se puede leer, intenta relacionarlo con el contexto colombiano y llenalo de forma inteligente, si no deja el campo vacío.
        - Deduce el género (M o F) según el nombre, usando contexto colombiano.
        - No agregues comentarios, encabezados ni texto antes o después de la tabla.
        """
    )
    response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[prompt] + imagenes,
    )
    print(response.text)
    return response.text

# --- Procesar imágenes ---
def proceso_digitacion(input_folder, output_excel):
    archivos = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


    if not archivos:
        print("❌ No se encontraron imágenes en la carpeta.")
        return


    # Seleccionar ROI una vez
    print("🖱️ Selecciona la región de interés en la primera imagen...")
    roi = seleccionar_roi_manual(archivos[0])
    print("✅ ROI seleccionada:", roi)


    todos_los_resultados = []


    for i in range(0, len(archivos), 5):
        batch = archivos[i:i + 5]


        # Recortar todas las imágenes del lote
        imagenes = [recortar_con_roi(p, roi) for p in batch]


        print(f"📦 Procesando lote {i // 5 + 1} con {len(imagenes)} imágenes...")
        texto = extraer_textos(imagenes)
        df_lote = texto_a_dataframe(texto)
        df_lote["lote"] = i // 5 + 1
        todos_los_resultados.append(df_lote)


        if i + 5 < len(archivos):
            print("⏳ Esperando 12 segundos para respetar el límite gratuito...")
            time.sleep(12)
        
        # Guardar backup del lote
        os.makedirs("backups", exist_ok=True)
        backup_path = os.path.join("backups", f"backup_lote_{i // 5 + 1}.xlsx")
        df_lote.to_excel(backup_path, index=False)
        print(f"💾 Backup guardado en {backup_path}")


    df_final = pd.concat(todos_los_resultados, ignore_index=True)
    df_final.to_excel(output_excel, index=False)
    print(f"✅ Resultados guardados en {output_excel}")

def seleccionar_roi_manual(path, max_width=1000, max_height=800):
    """
    Permite seleccionar manualmente la región de interés en la imagen con el mouse.
    Redimensiona la imagen si es muy grande, y ajusta las coordenadas al tamaño original.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")

    # Guardar tamaño original
    orig_h, orig_w = img.shape[:2]

    # Calcular escala si la imagen es muy grande
    scale_w = max_width / orig_w
    scale_h = max_height / orig_h
    scale = min(scale_w, scale_h, 1.0)  # No agrandar imágenes pequeñas

    # Redimensionar solo si es necesario
    if scale < 1.0:
        resized = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    else:
        resized = img.copy()

    # Mostrar y seleccionar ROI
    r = cv2.selectROI("Selecciona la región de interés", resized)
    cv2.destroyAllWindows()

    # Convertir coordenadas a tamaño original
    x, y, w, h = [int(v / scale) for v in r]
    return (x, y, x + w, y + h)


def recortar_con_roi(path, roi):
    """
    Recorta la imagen usando una ROI definida.
    """
    img = cv2.imread(path)
    x1, y1, x2, y2 = roi
    recorte = img[y1:y2, x1:x2]
    recorte_pil = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    return recorte_pil


if __name__ == "__main__":
    proceso_digitacion(input_folder, output_excel)