import gradio as gr
import spaces
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# --- CARGA DEL MODELO (Comentada por ahora) ---
# Cuando tu compañero te pase el modelo real, descomenta estas líneas:
# model = models.efficientnet_b0(weights='DEFAULT')
# model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- FUNCIÓN DE INFERENCIA (MOCK) ---
# El decorador @spaces.GPU es obligatorio para que arranque en ZeroGPU
@spaces.GPU
def triage(image):
    if image is None:
        return "Por favor, sube una imagen para analizar."
    
    # --- Lógica simulada ---
    # Aquí es donde irá el código real para procesar la imagen:
    # img_t = transform(image).unsqueeze(0)
    # with torch.no_grad():
    #     output = model(img_t)
    #     prob = torch.softmax(output, dim=1)
    
    # Simulamos un resultado para que veas cómo queda el diseño
    return "⚠️ RESULTADO DE PRUEBA:\n\nLa lesión presenta características que sugieren una revisión prioritaria por un dermatólogo.\n\n*(Este es un mensaje simulado para probar el diseño)*"

# --- INTERFAZ GRÁFICA ---
# Puedes cambiar el tema aquí: gr.themes.Soft(), gr.themes.Monochrome(), etc.
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 DermaTriage")
    gr.Markdown("Sube una foto de una lesión cutánea para obtener una evaluación preliminar de triaje.")
    
    with gr.Row():
        # Columna izquierda: Entrada de imagen y botón
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Foto de la lesión")
            btn = gr.Button("Analizar Imagen", variant="primary")
            
        # Columna derecha: Salida de texto
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Resultado del Triaje", 
                lines=6, 
                interactive=False
            )
    
    # Conectamos el botón a la función triage
    btn.click(fn=triage, inputs=input_img, outputs=output_text)
    
    # Aviso legal
    gr.Markdown("---")
    gr.Markdown("⚠️ **Aviso legal:** Esta es una demo educativa para el Máster en IA Avanzada. No es un dispositivo médico ni sustituye el diagnóstico de un dermatólogo.")

# Lanzar la app
if __name__ == "__main__":
    demo.launch()