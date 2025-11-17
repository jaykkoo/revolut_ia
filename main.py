import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

##############################################
# 1) MINI IA POUR PRÉDICTION / MANAGEMENT
##############################################

mini_name = "Qwen/Qwen2-0.5B-Instruct"
mini_tokenizer = AutoTokenizer.from_pretrained(mini_name)
mini_model = AutoModelForCausalLM.from_pretrained(
    mini_name, torch_dtype=torch.float16, device_map="auto"
)

def mini_ai_predict_fragments(user_prompt: str, num_layers: int):
    """
    La mini-IA prédit quelles couches du modèle seront les plus sollicitées.
    Simple version basée sur tokens + heuristiques.
    """

    system = (
        "Tu es un analyseur. Détermine quelles couches d'un LLM seront "
        "les plus actives pour répondre à cette instruction."
        "Renvoie une liste de numéros de couches pertinents."
    )

    prompt = f"{system}\nInstruction : {user_prompt}\nCouches pertinentes :"

    inputs = mini_tokenizer(prompt, return_tensors="pt").to(mini_model.device)
    out = mini_model.generate(**inputs, max_new_tokens=80, temperature=0.2)
    text = mini_tokenizer.decode(out[0], skip_special_tokens=True)

    # extraction naïve : chiffres trouvés dans la réponse
    import re
    nums = [int(x) for x in re.findall(r"\d+", text)]
    return [n for n in nums if n < num_layers]


##############################################
# 2) VRAM FRAGMENTÉE
##############################################

class FragmentedModelManager:
    """
    Gère un modèle LLM fragmenté en plusieurs couches sur CPU ↔ GPU.
    Inspiré des approches vLLM, DeepSpeed-Zero-Offload, ExLlama2.
    """

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.num_layers = len(self.layers)

        print(f"[INFO] Modèle contenant {self.num_layers} fragments (layers).")

        # Tout le monde sur CPU par défaut
        for layer in self.layers:
            layer.to("cpu")

        self.device = torch.device("cuda")
        self.loaded = {}

    def load_fragment(self, i: int):
        """Charge un fragment dans la VRAM ET ses buffers."""
        if i in self.loaded:
            return
        
        layer = self.layers[i]

        # Move weights
        layer.to(self.device)

        # Move all buffers (attention masks, rotary embeddings, etc.)
        for name, buf in layer.named_buffers(recurse=True):
            setattr(layer, name, buf.to(self.device))

        self.loaded[i] = True

    def unload_fragment(self, i: int):
        """Décharge un fragment en CPU."""
        if i not in self.loaded:
            return

        layer = self.layers[i]

        # Move weights
        layer.to("cpu")

        # Move buffers
        for name, buf in layer.named_buffers(recurse=True):
            setattr(layer, name, buf.to("cpu"))

        del self.loaded[i]


    def unload_all(self):
        """Vide toute la VRAM."""
        for i in list(self.loaded.keys()):
            self.unload_fragment(i)

    def prefetch(self, layer_ids):
        """
        Pré-charge certains fragments prévus par la mini-IA.
        """
        for i in layer_ids:
            self.load_fragment(i)

    def forward(self, input_ids, attention_mask=None):
        """
        Exécution couche par couche :
        Chargement dynamique + déchargement après utilisation
        """
        hidden = self.model.model.embed_tokens(input_ids.to(self.device))

        for i, layer in enumerate(self.layers):

            # Charger si pas déjà en VRAM
            self.load_fragment(i)

            # Exécuter
            hidden = layer(hidden)[0]

            # Décharger immédiatement
            self.unload_fragment(i)

        # head finale (non fragmentée)
        logits = self.model.lm_head(hidden)
        return logits


##############################################
# 3) CHARGEMENT DU MISTRAL 7B EN FRAGMENTS
##############################################

mistral_name = "mistralai/Mistral-7B-Instruct-v0.2"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_name)

# On charge TOUT en CPU en 8-bit
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_name,
    load_in_8bit=True,           # ou load_in_4bit=True
    device_map={"": "cpu"}       # important pour fragmentation
)

frag = FragmentedModelManager(mistral_model)


##############################################
# 4) PIPELINE COMPLET
##############################################

def generate_code(user_prompt: str):

    # Analyse mini IA
    predicted_layers = mini_ai_predict_fragments(
        user_prompt, num_layers=frag.num_layers
    )

    print("Couches prédites par la mini IA :", predicted_layers)

    # Préfetch des fragments les plus importants
    frag.prefetch(predicted_layers[:5])  # on limite

    # Construire prompt Mistral
    messages = [
        {"role": "system", "content": "Tu es expert DRF, génère du code propre et complet."},
        {"role": "user", "content": user_prompt}
    ]
    prompt = mistral_tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    )

    # Générer un token à la fois (démo)
    input_ids = prompt.to(frag.device)

    outputs = []
    for _ in range(200):
        logits = frag.forward(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        outputs.append(next_token.item())

    text = mistral_tokenizer.decode(outputs)
    return text


##############################################
# 5) EXEMPLE D’UTILISATION
##############################################

if __name__ == "__main__":
    out = generate_code("Créer une API DRF CRUD pour gérer un modèle Product(name, price, stock)")
    print("\n===== CODE DRF GÉNÉRÉ =====\n")
    print(out)
