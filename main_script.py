import os
import json
import gradio as gr
from huggingface_hub import HfApi, HfHubError

# Global variables for settings
settings = {}

# Load settings from external JSON
def load_settings():
    global settings
    try:
        with open("config.json", "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        # If config.json doesn't exist, set default values
        settings = {
            "model_directory": "",
            "output_directory": "",
            "huggingface_token": ""
        }

# Save settings to external JSON
def save_settings():
    global settings
    with open("config.json", "w") as f:
        json.dump(settings, f, indent=4)

# Function to extract metadata from the folder structure (author/modelname)
def generate_metadata_from_path(model_path):
    path_parts = model_path.strip("/").split("/")
    if len(path_parts) < 3:
        raise ValueError("Path must be in the form './author/modelname'.")
    
    author = path_parts[-2]  # Extract author from folder name
    modelname = path_parts[-1].split('.')[0]  # Extract model name before the first dot
    
    metadata = {
        "author": author,
        "model_name": modelname,
        "description": f"Auto-generated model card for {author}/{modelname}",
        "license": "Unknown",
        "format": "GGUF",
        "parameters": "Unknown",
        "architecture": "Unknown"
    }
    
    return metadata

# Function to interact with Hugging Face API to fetch model info
def fetch_huggingface_metadata(author, modelname):
    hf_api = HfApi()
    model_id = f"{author}/{modelname}"

    try:
        model_info = hf_api.model_info(model_id, token=settings.get("huggingface_token", None))
        hf_metadata = {
            "description": model_info.cardData.get("model_description", "Description from Hugging Face."),
            "license": model_info.license or "Unknown",
            "parameters": model_info.cardData.get("model_parameters", "Unknown"),
            "architecture": model_info.cardData.get("model_architecture", "Unknown"),
        }
        return hf_metadata
    except HfHubError as e:
        return None

# Function to read a README.md file if it exists and extract additional information
def parse_readme(readme_path, metadata):
    if os.path.exists(readme_path) and readme_path.endswith(".md"):
        with open(readme_path, 'r') as readme_file:
            readme_content = readme_file.read()
            
            # Update description with the first few lines of the README
            metadata["description"] = readme_content.split("\n")[0][:100] + "..."  # First 100 chars of README
    return metadata

# Function to resolve conflicts or prompt user for missing data via Gradio
def resolve_metadata_conflicts(hf_metadata, local_metadata):
    result_metadata = {}

    for key in hf_metadata:
        if hf_metadata[key] != local_metadata[key] and hf_metadata[key] != "Unknown":
            # Prompt the user to choose between Hugging Face and local metadata
            choice = gr.Interface(
                fn=lambda hf_value, local_value: hf_value if hf_value != "Unknown" else local_value,
                inputs=[
                    gr.Textbox(value=hf_metadata[key], label=f"Hugging Face: {key}"),
                    gr.Textbox(value=local_metadata[key], label=f"Local: {key}")
                ],
                outputs="text"
            )
            result = choice.launch(inline=True)
            result_metadata[key] = result
        else:
            result_metadata[key] = hf_metadata[key] if hf_metadata[key] != "Unknown" else local_metadata[key]
    
    return result_metadata

# Generate model card JSON file
def generate_model_card(model_path, output_dir, author, modelname):
    local_metadata = generate_metadata_from_path(model_path)
    
    # Check for README.md file in the model directory to enrich metadata
    readme_path = os.path.join(os.path.dirname(model_path), "README.md")
    local_metadata = parse_readme(readme_path, local_metadata)
    
    # Fetch Hugging Face metadata
    hf_metadata = fetch_huggingface_metadata(author, modelname)
    
    if hf_metadata:
        # Resolve conflicts between Hugging Face and local metadata
        metadata = resolve_metadata_conflicts(hf_metadata, local_metadata)
    else:
        metadata = local_metadata
    
    model_name = metadata["model_name"]
    output_file = os.path.join(output_dir, f"{model_name}.json")
    
    # Write the final resolved model card as a JSON file
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)
    
    return f"Model card for {model_name} saved to {output_file}"

# Main function to process the directory and create model cards
def process_models_directory(model_dir, output_dir):
    output_info = []
    
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".gguf"):  # Only process .gguf files
                model_path = os.path.join(root, file)
                author = model_path.split("/")[-3]
                modelname = model_path.split("/")[-2].split("-")[0]
                result = generate_model_card(model_path, output_dir, author, modelname)
                output_info.append(result)
    
    return "\n".join(output_info)

# Gradio Interface for "Generate Model Cards"
def gradio_interface(model_dir, output_dir):
    if not model_dir or not os.path.exists(model_dir):
        return "Error: Invalid model directory!"
    
    if not output_dir or not os.path.exists(output_dir):
        return "Error: Invalid output directory!"
    
    result = process_models_directory(model_dir, output_dir)
    return result

# Function to update settings from Gradio input
def update_settings(model_dir, output_dir, huggingface_token):
    global settings
    settings["model_directory"] = model_dir
    settings["output_directory"] = output_dir
    settings["huggingface_token"] = huggingface_token
    save_settings()
    return "Settings saved!"

# Load the settings when the script starts
load_settings()

# Create the Gradio app
iface = gr.Blocks()

with iface:
    gr.Markdown("# Model Card Generator with Settings")
    
    # Settings Section
    with gr.Row():
        gr.Markdown("### Settings")
        model_dir_input = gr.Textbox(label="Model Directory", value=settings["model_directory"])
        output_dir_input = gr.Textbox(label="Output Directory", value=settings["output_directory"])
        hf_token_input = gr.Textbox(label="Hugging Face Token", value=settings["huggingface_token"])
    
    save_button = gr.Button("Save Settings")
    
    # Generate Model Cards Section
    with gr.Row():
        gr.Markdown("### Generate Model Cards")
        generate_button = gr.Button("Generate Model Cards")
    
    output_display = gr.Textbox(label="Output", placeholder="Model card generation results will appear here...")
    
    # Actions
    save_button.click(
        fn=update_settings,
        inputs=[model_dir_input, output_dir_input, hf_token_input],
        outputs=output_display
    )
    
    generate_button.click(
        fn=gradio_interface,
        inputs=[model_dir_input, output_dir_input],
        outputs=output_display
    )

# Launch the app
iface.launch()
