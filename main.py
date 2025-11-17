import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer, ExLlamaV2Config, ExLlamaV2Generator

###############################################
# 1) MINI IA - pour optimiser la demande
###############################################

mini_name = "Qwen/Qwen2-0.5B-Instruct"
mini_tokenizer = AutoTokenizer.from_pretrained(mini_name)
mini_model = AutoModelForCausalLM.from_pretrained(
    mini_name, torch_dtype=torch.float16, device_map="auto"
)

def mini_ai_optimize(prompt):
    system = (
        "Réécris la demande de l'utilisateur pour un modèle spécialisé DRF. "
        "Transforme-la en instruction claire, concise et structurée."
    )

    prompt2 = f"{system}\n\nDemande utilisateur : {prompt}\n\nInstruction DRF optimisée :"
    inputs = mini_tokenizer(prompt2, return_tensors="pt").to(mini_model.device)

    out = mini_model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return mini_tokenizer.decode(out[0], skip_special_tokens=True)


###############################################
# 2) CONFIGURATION EXLLAMAV2
###############################################

model_path = "./models/mistral/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

config = ExLlamaV2Config(model_path)
config.max_seq_len = 4096                   # supporte 10k+ si besoin
config.gpu_split = [10.0]                   # 10 Go VRAM disponibles
config.compress_pos_emb = False

# Charger modèle ExLlamaV2
model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)

# Générateur ExLlamaV2
generator = ExLlamaV2Generator(
    model,
    tokenizer,
    max_seq_len=4096,
    max_gen_len=512
)

generator.settings.temperature = 0.2
generator.settings.top_p = 0.95
generator.settings.top_k = 50


###############################################
# 3) PIPELINE COMPLET
###############################################

def generate_code(prompt):
    print("Mini IA en cours...")
    optimized = mini_ai_optimize(prompt)
    print("Instruction optimisée =", optimized)

    final_prompt = (
        "Tu es un expert Django REST Framework. Génère du code propre, clair, "
        "commenté et fonctionnel. Utilise des sections : models.py, serializers.py, "
        "views.py, urls.py.\n\n"
        f"{optimized}"
    )

    print("Mistral 7B ExLlamaV2 en cours...")
    output = generator.generate(final_prompt)
    return output


###############################################
# 4) TEST
###############################################

if __name__ == "__main__":
    result = generate_code(
        "Créer une API DRF CRUD pour un modèle Product(name, price, stock)."
    )
    print("\n===== CODE DRF GÉNÉRÉ =====\n")
    print(result)
