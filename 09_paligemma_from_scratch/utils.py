from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple 
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads a Hugging Face model and tokenizer from local files.
    
    Args:
        model_path: Path to the directory containing model files
        device: Target device for model loading ('cpu', 'cuda', etc.)
    
    Returns:
        Tuple of (loaded model, tokenizer)
    """
    # Load the tokenizer with right-padding (important for attention masks)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    
    assert tokenizer.padding_side == "right"  # Ensure correct padding for causal attention

    # Find all model weight files (stored in safetensors format for safety and efficiency)
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Load all model weights into a single dictionary
    # safetensors is a safer alternative to pickle for storing weights
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    
    # Load model configuration from JSON
    # This contains architecture details like number of layers, hidden size, etc.
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Initialize model architecture with config and move to specified device
    model = PaliGemmaForConditionalGeneration(config).to(device)
    # Load weights into model (strict=False allows partial loading)
    model.load_state_dict(tensors, strict=False)

    # Ensure input/output embeddings share weights (common in language models)
    # This reduces model size and can improve performance
    model.tie_weights()

    return (model, tokenizer)
