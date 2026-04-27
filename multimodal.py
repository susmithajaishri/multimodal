import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, XLMRobertaTokenizer, XLMRobertaModel
import pytesseract

# ===============================
# 1️⃣ Setup & Robust Data Loading
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/dataset"

def load_and_standardize(file_name, sub_folder, img_folder):
    full_path = os.path.join(base_path, file_name)
    if not os.path.exists(full_path):
        print(f"❌ Skipping {file_name}: File not found at {full_path}")
        return pd.DataFrame()

    df = pd.read_excel(full_path)
    # Standardize column names
    df = df.rename(columns={'meme_id':'id', 'Image_id':'id', 'Level 1':'Level1', 'Level 2':'Level2'})
    df['id'] = df['id'].astype(str).str.replace(r'\.jpg|\.jpeg|\.png', '', regex=True, case=False)

    def find_image(img_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(base_path, sub_folder, img_folder, f"{img_id}{ext}")
            if os.path.exists(path): return path
        return None

    df["image_path"] = df["id"].apply(find_image)
    print(f"✅ {file_name}: Found {df['image_path'].notna().sum()} images.")
    return df

# Load and Merge (The fix for your KeyError)
train_parts = [
    load_and_standardize("Malayalam_Train_label.xlsx", "Train", "Malayalam_Train_images"),
    load_and_standardize("Tamil_Train_label.xlsx", "Train", "Tamil_Train_images")
]

# Filter out empty dataframes before concatenating
train_parts = [d for d in train_parts if not d.empty]
if not train_parts:
    raise ValueError("No data found! Check your Google Drive paths.")

train_df = pd.concat(train_parts).dropna(subset=['image_path']).reset_index(drop=True)

le1, le2 = LabelEncoder(), LabelEncoder()
train_df['L1_enc'] = le1.fit_transform(train_df['Level1'].astype(str))
train_df['L2_enc'] = le2.fit_transform(train_df['Level2'].astype(str))

# ===============================
# 2️⃣ Feature Extraction
# ===============================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
xlmr_model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device).eval()
xlmr_tok = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def get_features_into_memory(df):
    img_feats, txt_feats = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        try:
            img = Image.open(row['image_path']).convert("RGB")
            # Multilingual OCR (Tamil + Malayalam + English)
            text = pytesseract.image_to_string(img, lang='tam+mal+eng').strip()
            text = text if len(text) > 2 else "political meme"

            with torch.no_grad():
                # CLIP Image
                i_emb = clip_model.get_image_features(**clip_proc(images=img, return_tensors="pt").to(device))
                # XLM-R Text
                t_out = xlmr_model(**xlmr_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device))
                t_emb = t_out.last_hidden_state[:, 0, :]

            img_feats.append(i_emb.squeeze().cpu().numpy())
            txt_feats.append(t_emb.squeeze().cpu().numpy())
        except Exception as e:
            img_feats.append(np.zeros(512))
            txt_feats.append(np.zeros(768))
    return np.array(img_feats), np.array(txt_feats)

print("⚡ Processing Train Features...")
train_i, train_t = get_features_into_memory(train_df)

# ===============================
# 3️⃣ Model & Dataset
# ===============================


class PoliticalMemeModel(nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.stance_head = nn.Linear(512, n1)
        self.target_head = nn.Linear(512, n2)

    def forward(self, i, t):
        x = self.fusion(torch.cat((i, t), dim=1))
        return self.stance_head(x), self.target_head(x)

class FastDataset(Dataset):
    def __init__(self, i, t, l1=None, l2=None):
        self.i, self.t, self.l1, self.l2 = i, t, l1, l2
    def __len__(self): return len(self.i)
    def __getitem__(self, idx):
        return torch.tensor(self.i[idx], dtype=torch.float32), \
               torch.tensor(self.t[idx], dtype=torch.float32), \
               (torch.tensor(self.l1[idx]) if self.l1 is not None else 0), \
               (torch.tensor(self.l2[idx]) if self.l2 is not None else 0)

# ===============================
# 4️⃣ Training Loop
# ===============================
train_loader = DataLoader(FastDataset(train_i, train_t, train_df['L1_enc'].values, train_df['L2_enc'].values), batch_size=32, shuffle=True)
model = PoliticalMemeModel(len(le1.classes_), len(le2.classes_)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("🏋️ Training...")
for epoch in range(15):
    model.train()
    for i, t, l1, l2 in train_loader:
        i, t, l1, l2 = i.to(device), t.to(device), l1.to(device), l2.to(device)
        o1, o2 = model(i, t)
        loss = criterion(o1, l1) + criterion(o2, l2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"Epoch {epoch+1} Complete")

# ===============================
# 5️⃣ Test Extraction & Generate CSV
# ===============================

# --- STEP A: Load the Test Excel ---
print("📂 Loading Test Data...")
test_df = load_and_standardize("Tamil_Test_Data.xlsx", "Test", "Tamil_Test_images") # Adjust filename if Malayalam
if test_df.empty:
    print("⚠️ Check test filename! Trying fallback...")
    # Add your test loading logic here if different

# --- STEP B: Extract Test Features ---
print("⚡ Processing Test Features (276 images)...")
test_i, test_t = get_features_into_memory(test_df)

# --- STEP C: Final Inference ---
print("🏁 Final Inference...")
model.eval()
res = []

with torch.no_grad():
    for i in range(len(test_i)):
        # .float() solves the 'Double vs Float' error
        img_input = torch.tensor([test_i[i]]).to(device).float()
        txt_input = torch.tensor([test_t[i]]).to(device).float()

        o1, o2 = model(img_input, txt_input)

        res.append([
            le1.inverse_transform([torch.argmax(o1).item()])[0],
            le2.inverse_transform([torch.argmax(o2).item()])[0]
        ])

# --- STEP D: Save Result ---
test_df[['stance', 'target']] = res
test_df[['id', 'stance', 'target']].to_csv("submission.csv", index=False)

print("🎊 SUCCESS! 'submission.csv' is ready. Check the folder icon on the left!")
# --- STEP A: Load the Test Excel ---
print("📂 Loading Test Data...")

# 1. Use the path you copied here!
test_file_path = "/content/drive/MyDrive/dataset/Tamil_Test_label.xlsx"

# 2. Load the data
test_df = pd.read_excel(test_file_path)

# 3. Standardize IDs (removes .jpg extensions if present)
test_df = test_df.rename(columns={'meme_id':'id', 'Image_id':'id'})
test_df['id'] = test_df['id'].astype(str).str.replace(r'\.jpg|\.jpeg|\.png', '', regex=True, case=False)

# 4. Helper to find images in the Test folder
def find_test_image(img_id):
    for ext in ['.jpg', '.jpeg', '.png']:
        # IMPORTANT: Check if your folder is named 'Tamil_Test_images' or just 'Test_images'
        path = os.path.join(base_path, "Test", "Tamil_Test_images", f"{img_id}{ext}")
        if os.path.exists(path): return path
    return None

test_df["image_path"] = test_df["id"].apply(find_test_image)

# 5. Verify images were found
found_count = test_df['image_path'].notna().sum()
print(f"✅ Found {found_count} test images.")

if found_count == 0:
    print("❌ ERROR: No images found! Check if 'Tamil_Test_images' folder name is correct.")
else:
    # --- STEP B: Extract Test Features ---
    print("⚡ Extracting Features for 276 test memes...")
    test_i, test_t = get_features_into_memory(test_df)

    # --- STEP C: Final Inference ---
    print("🏁 Generating Predictions...")
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(len(test_i)):
            img_in = torch.tensor([test_i[i]]).to(device).float()
            txt_in = torch.tensor([test_t[i]]).to(device).float()
            o1, o2 = model(img_in, txt_in)
            res.append([
                le1.inverse_transform([torch.argmax(o1).item()])[0],
                le2.inverse_transform([torch.argmax(o2).item()])[0]
            ])

    # --- STEP D: Save Result ---
    test_df[['stance', 'target']] = res
    test_df[['id', 'stance', 'target']].to_csv("submission.csv", index=False)
    print("🎊 SUCCESS! 'submission.csv' is generated. Refresh the folder icon on the left to see it.")
import pandas as pd
check_df = pd.read_csv("submission.csv")

print(f"✅ Total Predictions: {len(check_df)}") # Must be 276
print(f"✅ Columns: {list(check_df.columns)}") # Must be ['id', 'stance', 'target']
print(check_df.head()) # Ensure stance/target are text (e.g., 'Troll'), not numbers
