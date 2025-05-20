import torch
from src.model import CNNModel
from src.dataset import get_dataloaders
from src.inference import predict_and_show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli ve verileri hazırla
model = CNNModel()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))

_, test_loader = get_dataloaders()

# Tahminleri göster
predict_and_show(model, test_loader, device=device, num_images=10)