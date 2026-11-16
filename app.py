from xml.parsers.expat import model

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

model = models.efficientnet_b0(weights='DEFAULT')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def triage(image):
    if image is None:
        return "Sube una imagen por favor."
    
    # Preprocesamiento
    img_t = transform(image).unsqueeze(0)
    
    # Inferencia (simulada por ahora para que no falle)
    with torch.no_grad():
        # En un modelo real aquí iría: output = model(img_t)
        # Por ahora devolvemos un texto fijo para probar la UI
        pass
    
    # Lógica de triaje placeholder (vuestra compañera/o la cambiará)
    return "⚠️ RESULTADO DE PRUEBA: La lesión parece sospechosa. (Esto es un mock hasta que integremos el modelo real)."

# --- INTERFAZ GRADIO ---
demo = gr.Interface(
    fn=triage,
    inputs=gr.Image(type="pil", label="Sube foto de la lesión"),
    outputs=gr.Text(label="Resultado del Triaje"),
    title="Triaje de Cáncer de Piel (Demo)",
    description="App en desarrollo para el Máster de IA. No es un diagnóstico médico."
)

if __name__ == "__main__":
    demo.launch()