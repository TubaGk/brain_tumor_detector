'''from src.dataset import get_dataloaders
from src.model import CNNModel
from src.train import train_model
from evaluate import evaluate_model, plot_results
import torch
import os


classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs("outputs", exist_ok=True)


train_loader, test_loader = get_dataloaders()
model = CNNModel()


model, acc_list, loss_list, val_acc_list,val_loss_list = train_model(model, train_loader, test_loader, num_epochs=10, device=device)#4 tane olucak


torch.save(model.state_dict(), "model_weights.pth")
print("Model ağırlıkları 'model_weights.pth' dosyasına kaydedildi.")


test_acc, cm, class_names = evaluate_model(model, test_loader, device=device, class_names=classes)


plot_results(acc_list, loss_list, val_acc_list,val_loss_list, cm, class_names, save_path="outputs/performance_graph.png")

print("Eğitim grafikleri 'outputs/performance_graph.png' dosyasına kaydedildi.")
'''
from src.model import CNNModel
from src.dataset import get_dataloaders
from src.inference import predict_and_show
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()


_, test_loader = get_dataloaders()

predict_and_show(model, test_loader, device=device, num_images=5)

