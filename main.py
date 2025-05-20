'''from src.dataset import get_dataloaders
from src.model import CNNModel
from src.train import train_model
from evaluate import evaluate_model, plot_results
import torch

# Sınıf isimleri ve cihaz seçimi
classes = ['no_tumor', 'pituitary', 'glioma', 'meningioma']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yükleyiciler ve model
train_loader, test_loader = get_dataloaders()
model = CNNModel()

# Modeli eğit
model, acc_list, loss_list = train_model(model, train_loader, test_loader, num_epochs=10, device=device)

# Ağırlıkları kaydet
torch.save(model.state_dict(), "model_weights.pth")
print("Model ağırlıkları 'model_weights.pth' dosyasına kaydedildi.")

# Modeli değerlendir
test_acc, cm, class_names = evaluate_model(model, test_loader, device=device, class_names=classes)

# Sonuçları görselleştir
plot_results(acc_list, loss_list, cm, class_names)

'''

from src.model import CNNModel 
from src.dataset import get_dataloaders
from src.inference import predict_and_show
import torch

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli oluştur ve kaydedilen ağırlıkları yükle
model = CNNModel()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()  # Değerlendirme moduna al

# Sadece test verisini al
_, test_loader = get_dataloaders()

# Tahminleri göster (örneğin 10 görsel için)
predict_and_show(model, test_loader, device=device, num_images=5)

