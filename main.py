from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

###########################################
# 1) MINI IA (Qwen 0.5B)
###########################################

mini_name = "Qwen/Qwen2-0.5B-Instruct"
mini_tokenizer = AutoTokenizer.from_pretrained(mini_name)
mini_model = AutoModelForCausalLM.from_pretrained(
    mini_name, torch_dtype=torch.float16, device_map="auto"
)

def mini_ai_optimize(prompt):
    system = (
        "Réécris la demande en instruction précise pour générer du code Django REST Framework. "
        "Inclue ce qu'il faut générer : model, serializer, views, urls. "
        "Structure la requête de manière claire."
    )

    prompt2 = f"{system}\n\nDemande utilisateur : {prompt}\n\nInstruction optimisée :"
    inputs = mini_tokenizer(prompt2, return_tensors="pt").to(mini_model.device)
    out = mini_model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return mini_tokenizer.decode(out[0], skip_special_tokens=True)


###########################################
# 2) MISTRAL v0.3 AVEC VLLM
###########################################

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    dtype="float16",            # automatic mixed precision
    gpu_memory_utilization=0.90 # fits in 10GB
)

sampling = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=512
)

###########################################
# 3) PIPELINE COMPLET
###########################################

def generate_code(user_prompt):
    print("Mini IA en cours...")
    optimized = mini_ai_optimize(user_prompt)
    print("Instruction optimisée :", optimized)

    final_prompt = (
        "Tu es un expert Django REST Framework. "
        "Génère du code DRF complet, structuré en sections :\n"
        "- models.py\n"
        "- serializers.py\n"
        "- views.py\n"
        "- urls.py\n"
        "Le code doit être exécutable.\n\n"
        f"{optimized}"
    )

    print("Mistral v0.3 en cours...")
    output = llm.generate([final_prompt], sampling)[0].outputs[0].text
    return output


###########################################
# TEST
###########################################

if __name__ == "__main__":
    code = generate_code(
        "Crée une API DRF CRUD pour des produits (name, price, stock)."
    )
    print("\n===== CODE DRF GÉNÉRÉ =====\n")
    print(code)
