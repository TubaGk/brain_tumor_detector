import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random

classes = ['no_tumor', 'pituitary', 'glioma', 'meningioma']

def predict_and_show(model, test_loader, device="cpu", num_images=10):
    model.eval()
    model.to(device)

    # Tüm test verisini bir kerede topla (önemli!)
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            all_images.extend(images)
            all_labels.extend(labels)

    # Rastgele indeks seç
    indices = random.sample(range(len(all_images)), num_images)

    for idx in indices:
        image = all_images[idx].unsqueeze(0).to(device)  # [1, C, H, W]
        label = all_labels[idx]

        output = model(image)
        _, pred = torch.max(output, 1)

        img_show = image.squeeze().cpu().permute(1, 2, 0)
        plt.imshow(img_show)
        plt.title(f"Tahmin: {classes[pred.item()]} | Gerçek: {classes[label]}")
        plt.axis('off')
        plt.show()
