import fitz  # PyMuPDF
import os
import shutil
from sklearn.model_selection import train_test_split

def extract_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf = fitz.open(pdf_path)
    
    for page_number in range(len(pdf)):
        page = pdf[page_number]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = os.path.join(output_folder, f"page{page_number+1}_img{img_index+1}.{image_ext}")
            
            with open(image_filename, "wb") as f:
                f.write(image_bytes)

# Extract images
extract_images_from_pdf("real_images.pdf", "data/temp_real")
extract_images_from_pdf("fake_images.pdf", "data/temp_fake")

# Split into train and validation sets
def split_train_val(temp_folder, train_folder, val_folder, val_ratio=0.2):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    images = [f for f in os.listdir(temp_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)
    
    for img in train_imgs:
        shutil.move(os.path.join(temp_folder, img), os.path.join(train_folder, img))
    for img in val_imgs:
        shutil.move(os.path.join(temp_folder, img), os.path.join(val_folder, img))
    
# Split real and fake images
split_train_val("data/temp_real", "data/train/real", "data/val/real")
split_train_val("data/temp_fake", "data/train/fake", "data/val/fake")
