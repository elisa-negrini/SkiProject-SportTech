import json

# Leggi il file JSON
with open('dataset/annotations/JP0007/train/annotations_jump7.json', 'r') as f:
    data = json.load(f)

# Ordina le immagini per nome del file originale
images_sorted = sorted(data['images'], key=lambda x: x['extra']['name'])

# Seleziona un'immagine ogni due
selected_images = images_sorted[::2]  # Prende elemento 0, 2, 4, 6...

# Crea un set con gli ID delle immagini selezionate
selected_image_ids = {img['id'] for img in selected_images}

# Filtra le annotazioni corrispondenti alle immagini selezionate
selected_annotations = [
    ann for ann in data['annotations'] 
    if ann['image_id'] in selected_image_ids
]

# Crea il nuovo dizionario con i dati filtrati
filtered_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'categories': data['categories'],
    'images': selected_images,
    'annotations': selected_annotations
}

# Salva il nuovo file JSON
with open('annotations_jump7_filtered.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"Immagini totali: {len(data['images'])}")
print(f"Immagini selezionate: {len(selected_images)}")
print(f"Annotazioni totali: {len(data['annotations'])}")
print(f"Annotazioni selezionate: {len(selected_annotations)}")
print("\nPrime 10 immagini selezionate:")
for img in selected_images[:10]:
    print(f"  - {img['extra']['name']} (ID: {img['id']})")