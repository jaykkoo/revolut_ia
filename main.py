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
        "Inclue : model, serializer, views, urls. "
        "Structure la requête de manière claire."
    )

    full_prompt = f"{system}\n\nDemande utilisateur : {prompt}\n\nInstruction optimisée :"
    inputs = mini_tokenizer(full_prompt, return_tensors="pt").to(mini_model.device)
    out = mini_model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return mini_tokenizer.decode(out[0], skip_special_tokens=True)


###########################################
# 2) MAIN ENTRY
###########################################

if __name__ == "__main__":

    ###########################################
    # 2) MISTRAL AWQ (vLLM)
    ###########################################

    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",          
        dtype="auto",                   # <- IMPORTANT FIX
        tokenizer_mode="mistral",
        gpu_memory_utilization=0.70
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
        print("\nMini IA en cours...")
        optimized = mini_ai_optimize(user_prompt)
        print("Instruction optimisée :", optimized)

        final_prompt = (
            "Tu es un expert Django REST Framework. "
            "Génère du code DRF complet, structuré en sections :\n"
            "- models.py\n"
            "- serializers.py\n"
            "- views.py\n"
            "- urls.py\n"
            "Le code doit être exécuté sans erreur.\n\n"
            f"{optimized}"
        )

        print("\nMistral AWQ en cours...")
        result = llm.generate([final_prompt], sampling)[0].outputs[0].text
        return result

    ###########################################
    # 4) TEST
    ###########################################

    code = generate_code(
        "Crée une API DRF CRUD pour des produits (name, price, stock)."
    )

    print("\n===== CODE DRF GÉNÉRÉ =====\n")
    print(code)
