import os
import csv
import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('./src/model/best.pth')
    model.eval()
    return model.to(device)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 替换为test.py的数值
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict_and_save(model, test_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        test_images = sorted(os.listdir(test_dir))
        for image_name in test_images:
            image_path = os.path.join(test_dir, image_name)
            image_tensor = preprocess_image(image_path)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = predicted.item()
            writer.writerow([os.path.splitext(image_name)[0], prediction])

def main():
    model_path = './src/model.pth'
    test_dir = './testdata'
    output_csv = './cla_pre.csv'
    model = load_model(model_path)
    predict_and_save(model, test_dir, output_csv)

if __name__ == '__main__':
    main()
