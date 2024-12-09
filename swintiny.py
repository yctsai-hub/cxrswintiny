### By Porter
import torch
import numpy as np
import pickle
import json
import argparse
import os
from sklearn import metrics
from PIL import Image
from cxrclip.data.data_utils import load_tokenizer, load_transform, transform_image
from cxrclip.model import build_model

parser = argparse.ArgumentParser(description='optional arguments')
parser.add_argument("--model_path", type=str, required=True, help="model path")
parser.add_argument('--image_path', type=str, required=True, help='image path')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = os.path.join(args.model_path)
if not os.path.exists(model_path):
    raise ValueError(f"model path '{args.model_path}' not found")

image_path = args.image_path
if not os.path.exists(image_path):
    raise ValueError(f"image path '{args.image_path}' not found")

print("input image: ", image_path)

### load ckpt config ###
ckpt = torch.load(model_path, map_location="cpu")
ckpt_config = ckpt["config"]

ckpt_config['tokenizer']['cache_dir'] = "/home/richardlin/home/porter/cxr-clip/cache/tokenizer"
ckpt_config['model']['text_encoder']['cache_dir'] = "/home/richardlin/home/porter/cxr-clip/cache/text_encoder"

print(ckpt_config)

### load tokenizer ###
tokenizer = load_tokenizer(**ckpt_config["tokenizer"])
print(tokenizer)

### load model ###
model = build_model(
    model_config=ckpt_config["model"], loss_config=ckpt_config["loss"], tokenizer=tokenizer
)
model = model.to(device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

### load texts from openi ###
# with open('openi_texts.json', 'r', encoding='utf-8') as f:
#     texts = json.load(f)

with open("texts.json", "r") as file:
    texts = json.load(file)

print(texts)
text_tokens = []
for t in texts:
    text_tokens.append(tokenizer(t, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"]))
print(np.array(text_tokens).shape)

### load image ###
image_transforms = load_transform(split="test", transform_config=ckpt_config["transform"])
image = Image.open(image_path).convert("RGB")
image = transform_image(image_transforms, image, normalize="huggingface")
print(image.shape)

### get image and text features ###
image = image.unsqueeze(0)
image_embeddings = model.encode_image(image.to(device))
image_embeddings = model.image_projection(image_embeddings) if model.projection else image_embeddings
image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True) # normalize
image_embeddings = image_embeddings.detach().cpu().numpy()

text_embeddings = []
with torch.no_grad():
    for tt in text_tokens:
        text_emb = model.encode_text(tt.to(device))
        text_emb = model.text_projection(text_emb) if model.projection else text_emb
        text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True) # normalize
        text_embeddings.append(text_emb.cpu())
        torch.cuda.empty_cache()
# with open('openi_embs.pkl', 'wb') as f:
#     pickle.dump(text_embs, f)

# with open('openi_embs.pkl', 'rb') as f:
#     text_embeddings = pickle.load(f)

# image_embeddings = np.concatenate(image_embeddings, axis=0)
if len(text_embeddings) > 0:
    text_embeddings = np.concatenate(text_embeddings, axis=0)

### retrieval_image_text ###
identical_text_set = []
idx2label = {}
identical_indexes = []
for i, text in enumerate(texts):
    if text not in identical_text_set:
        identical_text_set.append(text)
        identical_indexes.append(i)
        idx2label[i] = len(identical_text_set) - 1
    else:
        idx2label[i] = identical_text_set.index(text)

identical_text_embedding = text_embeddings[identical_indexes]

similarities = metrics.pairwise.cosine_similarity(image_embeddings, identical_text_embedding)  # n x m
# similarities = (similarities + 1) / 2
# similarities = similarities ** 4
# similarities = similarities * 100
print(similarities.shape)

max_idx = np.argmax(similarities)
max_i, max_j = np.unravel_index(max_idx, similarities.shape)

print("--------------------------------------------------------------")
print("##### Result #####")
print(f"Maximum similarity: {similarities[max_i, max_j]}")
print(f"Text with highest similarity: {identical_text_set[max_j]}")
print("--------------------------------------------------------------")
print("##### Sorted by similarity in descending order #####")
similarity_list = []
for i in range(similarities.shape[0]):
    for j in range(similarities.shape[1]):
        similarity_list.append((similarities[i, j], i, j))

sorted_similarities = sorted(similarity_list, key=lambda x: x[0], reverse=True)

for similarity_score, i, j in sorted_similarities:
    corresponding_text = identical_text_set[j]
    print(f"similarity = {similarity_score:.4f}, text: {corresponding_text}")
