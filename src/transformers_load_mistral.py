import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.quantization as quantization

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN_v03")

def load_mistral():
    # Step 1: Confirm MPS availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Step 2: Set MPS as the default device
    torch.set_default_device("mps")

    # Step 3: Load Mistral model and tokenizer
    #model_id = "mistralai/Mistral-7B-v0.1"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3" # NEWEST MODEL REQUIRES USER TOKEN.

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HUGGINGFACE_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HUGGINGFACE_TOKEN, 
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    
    return model, tokenizer

def apply_qat(model):
    # Step 4: Apply Quantization-Aware Training (QAT)
    model.train()

    # Set the quantization configuration for the model
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')  # Or 'qnnpack' for mobile devices
    
    # Prepare the model for QAT
    quantization.prepare_qat(model, inplace=True)

    return model

def train_qat(model, tokenizer):
    # This is just a skeleton. You would need a proper training loop here.
    # Example: Fine-tuning the model on your dataset (not shown in this code).
    pass

def convert_and_save_quantized_model(model):
    # Step 5: Convert the model to a quantized version after QAT
    quantized_model = quantization.convert(model, inplace=False)

    # Save the quantized model
    quantized_model.save_pretrained("./quantized_mistral_model")

def main():
    # Step 1: Load Mistral model and tokenizer
    model, tokenizer = load_mistral()

    # Step 2: Apply Quantization-Aware Training (QAT)
    model = apply_qat(model)
    
    # Step 3: Fine-tune the model (dummy step here for illustration)
    train_qat(model, tokenizer)

    # Step 4: Convert and save the quantized model after QAT
    convert_and_save_quantized_model(model)

    print("Mistral model with QAT has been trained and saved!")

    # Step 5: Test the model (Inference)
    prompt = "Explain the process of quantization in machine learning."
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    outputs = model.generate(**inputs, max_new_tokens=100)

    # Display the output
    print("\nGenerated Response:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
