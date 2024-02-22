# Load original model
from models import ViT as SsdViT
import torch

args = {
    'net': 'ViT',
    'weight_path': 'checkpoint/ViT/Tuesday_09_January_2024_14h_53m_51s/ViT-Cifar100-8-best.pth',
    'dataset': 'Cifar100',
    'classes': 100,
    'gpu': True,  # This is set because of the '-gpu' flag
    'b': 64,  # Default value as not specified in the command
    'warm': 1,  # Default value as not specified in the command
    'lr': 0.1,  # Default value as not specified in the command
    'method': 'baseline',
    'forget_class': 'rocket',
    'epochs': 1,  # Default value as not specified in the command
    'seed': 123,  # From the 'seed=123' in the bash script
}

ssd_model = SsdViT(args["classes"])
ssd_weights = torch.load(args["weight_path"])
ssd_model.load_state_dict(ssd_weights)

# Convert to huggingface model
from transformers import ViTConfig, ViTForImageClassification

hf_config = ViTConfig.from_pretrained("Ahmed9275/Vit-Cifar100")
hf_model = ViTForImageClassification(hf_config)

hf_weights = {}
for key, val in ssd_weights.items():
    if "pooler" in key:
        print("Skipping:", key)
        continue
    if "base" in key:
        hf_weights["vit" + key[4:]] = val
        continue
    if "final" in key:
        hf_weights["classifier" + key[5:]] = val
        continue
    print("unknown key:", key)
    hf_weights[key] = val

hf_model.load_state_dict(hf_weights)


# Save as a new state dict
new_model_dir = "./checkpoint/vit_cifar100_hf"
hf_model.save_pretrained(new_model_dir)