import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base', device_map=None)
        self.fc = nn.Linear(768, 5)
    
    def forward(self, ids, mask, token_type_ids=None):
        _, features = self.roberta(ids, attention_mask=mask, return_dict=False)
        output = self.fc(features)
        output = torch.sigmoid(output)
        return output

model = BERTClass()
model.to(device)

state_dict = torch.load("model.bin", map_location=device)

# Xóa position_ids
for k in list(state_dict.keys()):
    if "position_ids" in k:
        state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

labels = ["anger", "joy", "sadness", "love", "fear"]

st.title("Multi-label Emotion Detection (5 class)")
text_input = st.text_area("Nhập câu:")

if st.button("Dự đoán"):
    if text_input.strip() == "":
        st.warning("Nhập câu đi cậu!")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            probs = model(input_ids, attention_mask).cpu().numpy()[0]

        threshold = 0.5
        pred_labels = [labels[i] for i, p in enumerate(probs) if p >= threshold]

        st.success("Những cảm xúc dự đoán:")
        st.write(pred_labels if pred_labels else "Không có cảm xúc nào vượt threshold")

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color='skyblue')
        ax.set_ylim(0,1)
        ax.set_ylabel("Xác suất")
        st.pyplot(fig)
