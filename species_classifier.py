import torch

# --- TRUC PENTRU REZOLVAREA ERORII DE SECURITATE PYTORCH ---
_orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = patched_load

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import timm
import os
from PIL import Image

from main_utils import download_dataset
from dotenv import load_dotenv

load_dotenv()
if not os.path.exists('data'):
    print("Folderul 'data' nu a fost găsit. Se începe descărcarea setului de date...")
    download_dataset()


# 1. Importăm transformările deja făcute de colegul tău în transformations.py
from transformations import build_cpu_training_transform, build_cpu_testing_transform

# 2. Dataset special pentru specia ta (citim din metadata.csv pe care îl aveți deja)
class SpeciesDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"data/{row['path']}" 
        label = row['species_label']
        
        # Deschidem imaginea
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 3. Pregătim datele
df = pd.read_csv('data/metadata.csv') 

# Curățăm numele speciilor (le facem litere mici și scoatem spațiile ca să fim siguri de potrivire)
df['species'] = df['species'].astype(str).str.lower().str.strip()

# Transformăm numele speciilor în cifre: 0, 1, 2
species_mapping = {'lynx': 0, 'salamander': 1, 'turtle_caretta': 2}
df['species_label'] = df['species'].map(species_mapping)

# FOARTE IMPORTANT: Ștergem rândurile care nu s-an potrivit (dacă există alte specii)
df = df.dropna(subset=['species_label'])
df['species_label'] = df['species_label'].astype(int)

print(f"Număr total de imagini valide găsite: {len(df)}")
print("Distribuția speciilor în dataset:\n", df['species'].value_counts())

# Împărțim datele: 80% antrenare, 20% validare
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Activăm transformările din proiectul vostru
train_transform = build_cpu_training_transform(image_size=288)
val_transform = build_cpu_testing_transform(image_size=288)

train_dataset = SpeciesDataset(train_df, transform=train_transform)
val_dataset = SpeciesDataset(val_df, transform=val_transform)

# --- MODIFICARE CRUCIALĂ PENTRU CPU: Scădem batch_size de la 32 la 4 ---
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# =====================================================================
# AICI ESTE PASUL 3 DE CARE ÎNTREBAI (Definirea Modelului cu num_classes=3)
# =====================================================================

# timm va lua automat modelul vostru de bază și îi va atașa stratul final de clasificare pentru 3 clase
model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-CNN-288', pretrained=True, num_classes=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Funcțiile clasice de antrenare pentru clasificare
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 4. Loop-ul de antrenare (pornim cu 3 epoci de test)
epochs = 3
print(f"Începe antrenarea pe dispozitivul: {device}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)  # Outputs va avea dimensiunea [batch_size, 3]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoca {epoch+1}/{epochs} - Loss mediu: {running_loss/len(train_loader):.4f}")

print("Antrenare finalizată! Începe evaluarea pe datele de validare...")

# 5. EVALUAREA PE DATELE DE VALIDARE
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Luăm specia cu cel mai mare scor
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n>>> Acuratețea finală pe datele de validare: {accuracy:.2f}% <<<")

# 6. SALVAREA MODELULUI PENTRU COLEGUL TĂU
checkpoint_path = "species_classifier_weights.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Succes! Fișierul '{checkpoint_path}' a fost salvat și este gata de trimis.")