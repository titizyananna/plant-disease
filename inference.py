import torch
import torch.nn.functional as F
from PIL import Image
from model import DiseaseClassifier
from torchvision import transforms
from huggingface_hub import hf_hub_download

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(repo_id="withtwon/plant-disease-model"):

    checkpoint_path = hf_hub_download(repo_id=repo_id,filename="best_model_2.pth")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    disease2idx = ckpt["disease2idx"]
    plant2idx = ckpt["plant2idx"]

    model = DiseaseClassifier(
        len(disease2idx),
        len(plant2idx)
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx2disease = {v: k for k, v in disease2idx.items()}

    return model, idx2disease

def predict_image(image_path, model, idx2disease):
    img = Image.open(image_path).convert("RGB")
    img = VAL_TF(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(img, lambda_=0.0)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = probs.topk(5)

    return [
        (idx2disease[i.item()], float(p.item()))
        for p, i in zip(top5.values, top5.indices)
    ]