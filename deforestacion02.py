# Importações
import os

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import gradio as gr
import random
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10 
IMG_SIZE = 128
DATA_DIR = "C:/Users/Rickson/Documents/Rickson/base/dataset"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# Função para equalização do histograma (usando OpenCV)
def equalize_hist(img):
    img = np.array(img)
    # Equalizar cada canal separadamente para imagens coloridas
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    elif len(img.shape) == 2:  # Para imagens em tons de cinza
        img = cv2.equalizeHist(img)
    return Image.fromarray(img)

# Transforms com equalização e normalização
def crop_right_half_safe(img):
    width, height = img.size
    if width >= 2 * height:  # checa se há duas metades
        img = img.crop((width // 2, 0, width, height))  # usa apenas metade direita
    return img

transform = transforms.Compose([
    transforms.Lambda(crop_right_half_safe),      # corta a imagem
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # redimensiona
    transforms.Lambda(equalize_hist),             # equaliza histograma
    transforms.ToTensor(),                        # converte para tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # normaliza
])


# Carregar dados
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

labels = sorted(os.listdir(DATA_DIR))

# Função para exibir um exemplo por classe
def show_examples(data_dir, class_names):
    plt.figure(figsize=(12, 6))
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        image_files = [f for f in os.listdir(class_path) if f.endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            continue
        img_file = random.choice(image_files)
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))

        # Aplica a equalização para visualização
        img_equalized = equalize_hist(img)

        plt.subplot(2, len(class_names), idx + 1)
        plt.imshow(img)
        plt.title(f"Ori: {class_name}")
        plt.axis("off")

        plt.subplot(2, len(class_names), idx + 1 + len(class_names))
        plt.tight_layout(w_pad=5.0)
        plt.imshow(img_equalized)
        plt.title(f"Eq: {class_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Exibir as imagens originais e equalizadas
show_examples(DATA_DIR, labels)

# Definir o modelo
weights = ResNet18_Weights.DEFAULT 
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model = model.to(device)

labels = sorted(os.listdir(DATA_DIR))

# Funções de treino e validação
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
val_accuracies = []

val_losses = []

def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_train_loss = 0
        model.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        total_val_loss = 0
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dl)
        acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(acc)

        print(f"[{epoch + 1:02d}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")
        torch.cuda.empty_cache()

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def plot_training_curves():
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='--')
    plt.plot(epochs, val_losses, label='Val Loss', color='red')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='green')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix():
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(xticks_rotation=45)
    plt.title("Matriz de Confusão")
    plt.show()

def report_metrics():
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Relatório por Classe:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

train()
evaluate()
plot_training_curves() 
plot_confusion_matrix()
report_metrics()


# Função de predição para o Gradio (MODIFICADA)
def predict(image_passada, image_presente):
    img_passada = transform(image_passada).unsqueeze(0).to(device)
    img_presente = transform(image_presente).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output_passada = model(img_passada)
        output_presente = model(img_presente)
        prob_passada = torch.nn.functional.softmax(output_passada, dim=1)
        prob_presente = torch.nn.functional.softmax(output_presente, dim=1)
        conf_passada, pred_passada = torch.max(prob_passada, 1)
        conf_presente, pred_presente = torch.max(prob_presente, 1)
        label_passada = CLASS_NAMES[pred_passada]
        label_presente = CLASS_NAMES[pred_presente]

        report = f"Imagem Passada: {label_passada} ({conf_passada.item():.2f})\n"
        report += f"Imagem Presente: {label_presente} ({conf_presente.item():.2f})\n\n"

        if label_passada != label_presente:
            report += f"Mudança Detectada: Área passou de {label_passada} para {label_presente}."
        else:
            report += "Nenhuma mudança significativa detectada."

        return report

# Interface Gradio (MODIFICADA)
gr.Interface(fn=predict,
             inputs=[gr.Image(type="pil", label="Imagem Passada"),
                     gr.Image(type="pil", label="Imagem Presente")],
             outputs="text",
             title="Detector de Mudanças em Terreno"
             ).launch(share=True)