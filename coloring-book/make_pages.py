

import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Configuration
image_dir = "./photos"
prompt = "A stylized black-and-white line art illustration of a natural scene with flowers and animals, suitable for a coloring book."
output_suffix = "-generated"
depth_suffix = "-depth"
html_file = os.path.join(image_dir, "preview.html")
image_size = 512
controlnet_strength = 0.3
guidance_scale = 15

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load ControlNet (depth version)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float32 if device != torch.device("cuda") else torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float32 if device != torch.device("cuda") else torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Load MiDaS depth model
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = depth_model.to(device)

def get_depth_map(input_path):
    image = Image.open(input_path).convert("RGB").resize((image_size, image_size))
    pixel_values = depth_feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = depth_model(pixel_values)
        depth = outputs.predicted_depth[0]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_image = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_image)

# Store HTML rows
html_rows = []

# Process images
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and output_suffix not in filename:
        input_path = os.path.join(image_dir, filename)
        base, _ = os.path.splitext(filename)
        depth_path = os.path.join(image_dir, f"{base}{depth_suffix}.png")
        output_path = os.path.join(image_dir, f"{base}{output_suffix}.png")

        print(f"Processing: {filename}")
        depth_image = get_depth_map(input_path)
        depth_image.save(depth_path)

        result = pipe(
            prompt=prompt,
            image=depth_image,
            num_inference_steps=30,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_strength
        )

        result.images[0].save(output_path)

        html_rows.append(f"""
        <tr>
            <td><img src="{filename}" width="256"></td>
            <td><img src="{os.path.basename(depth_path)}" width="256"></td>
            <td><img src="{os.path.basename(output_path)}" width="256"></td>
        </tr>
        """)

# Write HTML
with open(html_file, "w") as f:
    f.write(f"""
    <html>
    <head>
        <title>ControlNet Depth Results</title>
        <style>
            table {{ border-collapse: collapse; }}
            td {{ padding: 10px; text-align: center; }}
            img {{ border: 1px solid #aaa; }}
        </style>
    </head>
    <body>
        <h1>Depth Map ControlNet Comparison</h1>
        <table>
            <tr><th>Original</th><th>Depth Map</th><th>Generated</th></tr>
            {''.join(html_rows)}
        </table>
    </body>
    </html>
    """)
    print(f"HTML preview written to: {html_file}")