from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Moves all input tensors to the specified device (CPU/GPU/MPS).
    
    Args:
        model_inputs: Dictionary containing input tensors
        device: Target device for computation
    
    Returns:
        Dictionary with all tensors moved to specified device
    """
    # Dictionary comprehension to move each tensor to device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(
    processor: PaliGemmaProcessor, 
    prompt: str, 
    image_file_path: str, 
    device: str 
):
    """
    Prepares inputs for the PaliGemma model by processing both text and image.
    
    Args:
        processor: Handles tokenization and image preprocessing
        prompt: Text input to condition the generation
        image_file_path: Path to the image file
        device: Target device for tensors
    
    Returns:
        Dictionary containing:
        - input_ids: Tokenized text
        - attention_mask: Mask for padding
        - pixel_values: Processed image tensors
    """
    # Load and preprocess image
    image = image.open(image_file_path)
    images = [image]                  # Batch of size 1
    prompts = [prompt]                # Batch of size 1
    
    # Process both text and image through processor
    # This handles:
    # 1. Text tokenization
    # 2. Image resizing and normalization
    # 3. Creating attention masks
    model_inputs = processor(text=prompts, images=images)
    
    # Move processed inputs to target device
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Performs autoregressive text generation given a prompt and image input.
    
    Args:
        model: The PaliGemma model
        processor: Handles tokenization and image processing
        device: Device to run inference on
        prompt: Text prompt to condition generation
        image_file_path: Path to input image
        max_tokens_to_generate: Maximum sequence length
        temperature: Controls randomness in sampling
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling vs greedy decoding
    """
    # Process inputs (tokenize text and encode image)
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]              # Tokenized text
    attention_mask = model_inputs["attention_mask"]    # Mask for padding
    pixel_values = model_inputs["pixel_values"]        # Processed image

    # Initialize cache for storing key/value pairs (speeds up generation)
    kv_cache = KVCache()

    # Setup generation tracking
    stop_token = processor.tokenizer.eos_token_id     # Token indicating end of sequence
    generated_tokens = []                             # Store generated tokens

    # Autoregressive generation loop
    for _ in range(max_tokens_to_generate):
        # Get model predictions
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]                # Update cached key/values
        next_token_logits = outputs["logits"][:, -1, :]  # Get logits for next token

        # Choose next token either through sampling or greedy selection
        if do_sample:
            # Temperature scaling followed by nucleus sampling
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy selection (take highest probability token)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        assert next_token.size() == (1,1)  # Ensure single token selected
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        
        # Check for end of sequence
        if next_token.item() == stop_token:
            break
            
        # Prepare inputs for next iteration
        input_ids = next_token.unsqueeze(-1)  # Use new token as input
        # Extend attention mask for new token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1,1), device=input_ids.device)], dim=-1
        )
    
    # Combine all generated tokens and decode to text
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print the complete response (prompt + generated text)
    print(prompt + decoded)

def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Implements nucleus (top-p) sampling for text generation.
    
    Nucleus sampling keeps the smallest set of tokens whose cumulative probability
    exceeds p, then samples from this set. This helps balance diversity and quality
    by dynamically adjusting the sampling pool size.

    Args:
        probs: Token probabilities (batch_size, vocab_size)
        p: Probability threshold (e.g., 0.9 means sample from top 90% cumulative prob)
    """
    # Sort probabilities in descending order and keep track of original indices
    # probs_sort: sorted probabilities
    # probs_idx: original vocabulary indices
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Calculate cumulative probabilities
    # e.g., if probs = [0.5, 0.3, 0.2], cumsum = [0.5, 0.8, 1.0]
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Create mask for tokens beyond the top-p threshold
    # Subtract probs_sort to exclude current token from cumsum comparison
    # e.g., if p=0.9, mask tokens where (cumsum - prob) > 0.9
    mask = probs_sum - probs_sort > p

    # Zero out probabilities for tokens beyond the nucleus
    probs_sort[mask] = 0.0

    # Renormalize probabilities to sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample a token index from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # Map sampled index back to original vocabulary index
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token




def main(
    model_path: str = None,          # Path to the model weights and config
    prompt: str = None,              # Text prompt for the model
    image_file_path: str = None,     # Path to the input image
    max_tokens_to_generate: int = 100,  # Maximum length of generated text
    temperature: float = 0.8,        # Controls randomness (higher = more random)
    top_p: float = 0.9,             # Nucleus sampling threshold
    do_sample: bool = False,         # Whether to sample or use greedy decoding
    only_cpu: bool = False,          # Force CPU usage even if GPU is available
):
    """
    Main entry point for running inference with PaliGemma model.
    Handles device selection, model loading, and generation setup.
    """
    # Device selection logic
    device = "cpu"
    if not only_cpu:
        # Try to use GPU (CUDA) or Apple Silicon (MPS) if available
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    
    print("Device in use: ", device)

    # Load model and tokenizer from specified path
    print(f"loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()  # Set model to evaluation mode

    # Get vision-related config for processor setup
    num_image_tokens = model.config.vision_config.num_image_tokens  # Number of tokens for image representation
    image_size = model.config.vision_config.image_size             # Required image dimensions
    # Initialize processor for handling both text and image inputs
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Run inference with gradient computation disabled
    print("Running inference ...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

if __name__ == "__main__":
    fire.Fire(main)