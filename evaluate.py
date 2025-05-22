import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device="cpu", class_names=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, class_names


def plot_results(train_acc, train_loss, val_acc=None, val_loss=None, cm=None, class_names=None, save_path="training_results.png"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    axs[0].plot(train_acc, label='Train Accuracy (%)')
    axs[0].plot(train_loss, label='Train Loss')

    if val_acc is not None:
        axs[0].plot(val_acc, label='Val Accuracy (%)', linestyle='--')
    if val_loss is not None:
        axs[0].plot(val_loss, label='Val Loss', linestyle='--')

    axs[0].set_title('Model Performansı')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Değer')
    axs[0].legend()
    axs[0].grid(True)


    if cm is not None and class_names is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axs[1], cmap=plt.cm.Blues)
        axs[1].set_title("Confusion Matrix")
    else:
        axs[1].axis("off")
        axs[1].text(0.5, 0.5, "Confusion Matrix Verisi Yok", ha='center', va='center')

    plt.tight_layout()


    plt.savefig(save_path)
    print(f"Grafik başarıyla kaydedildi: {save_path}")

    plt.show()
