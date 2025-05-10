# Importações
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import gradio as gr
import random
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
IMG_SIZE = 256
DATA_DIR = "C:/Users/Rickson/Documents/Rickson/base/dataset"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# Nova função para segmentar a floresta (Otsu's method)
def segment_forest(image):
    img_np = np.array(image.convert('L'))  
    _, mask = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# Função para calcular a área da floresta (em pixels)
def calculate_forest_area(mask):
    return np.sum(mask == 255)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      
    transforms.ToTensor(),                        
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
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

        
        plt.subplot(2, len(class_names), idx + 1)
        plt.imshow(img)
        plt.title(f"Ori: {class_name}")
        plt.axis("off")

        
    plt.tight_layout()
    plt.show()

# Exibir as imagens originais e equalizadas
show_examples(DATA_DIR, labels)

# Definir o modelo
weights = ResNet50_Weights.DEFAULT  
model = resnet50(weights=weights)  
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
    best_val_loss = float('inf')
    patience = 5 
    trigger = 0
    best_model_state = None

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

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  
                break

        torch.cuda.empty_cache()

    return model 

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
plot_confusion_matrix()
report_metrics()


def predict(image_passada, image_presente):
    img_passada = transform(image_passada).unsqueeze(0).to(device)
    img_presente = transform(image_presente).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output_passada = model(img_passada)
        output_presente = model(img_presente)
        prob_passada = F.softmax(output_passada, dim=1)
        prob_presente = F.softmax(output_presente, dim=1)

        # --- Análise da Imagem Passada ---
        topk_values_passada, topk_indices_passada = torch.topk(prob_passada, 3, dim=1)
        max_conf_passada = topk_values_passada[0, 0].item()
        conf_diff_passada = topk_values_passada[0, 0].item() - topk_values_passada[0, 1].item()

        if max_conf_passada > 0.6 and conf_diff_passada > 0.2:
            label_passada = CLASS_NAMES[topk_indices_passada[0, 0]]
        else:
            label_passada = "Indefinida"

        # --- Análise da Imagem Presente ---
        topk_values_presente, topk_indices_presente = torch.topk(prob_presente, 3, dim=1)
        max_conf_presente = topk_values_presente[0, 0].item()
        conf_diff_presente = topk_values_presente[0, 0].item() - topk_values_presente[0, 1].item()

        if max_conf_presente > 0.6 and conf_diff_presente > 0.2:
            label_presente = CLASS_NAMES[topk_indices_presente[0, 0]]
        else:
            label_presente = "Indefinida"

        report = f"Imagem Passada: {label_passada} ({max_conf_passada:.2f})\n"
        report += f"Imagem Presente: {label_presente} ({max_conf_presente:.2f})\n\n"

        # --- Segmentação e Cálculo da Área ---
        mask_passada = segment_forest(image_passada)
        mask_presente = segment_forest(image_presente)
        area_passada = calculate_forest_area(mask_passada)
        area_presente = calculate_forest_area(mask_presente)
        area_diff = area_passada - area_presente

        report += f"Área de floresta (passada): {area_passada} pixels\n"
        report += f"Área de floresta (presente): {area_presente} pixels\n"
        report += f"Mudança na área de floresta: {area_diff} pixels\n"

        if label_passada != label_presente and "Indefinida" not in [label_passada, label_presente]:
            report += f"Mudança Detectada: Área passou de {label_passada} para {label_presente}."
        elif "Indefinida" in [label_passada, label_presente]:
            report += "Mudança Indefinida Detectada."
        else:
            report += "Nenhuma mudança significativa detectada."

        return report

# Interface Gradio (MODIFICADA)
gr.Interface(fn=predict,
             inputs=[gr.Image(type="pil", label="Imagem Passada"),
                     gr.Image(type="pil", label="Imagem Presente")],
             outputs="text",
             title="Detector de Mudanças em Terreno").launch(share=True)