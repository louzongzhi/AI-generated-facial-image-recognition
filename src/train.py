from data import RealFakeDataLoader


data_path = 'data'
cropSize = 256
batch_size = 32
num_threads = 4

data_loader = RealFakeDataLoader(data_path, cropSize, batch_size, num_threads)

for i, (img, label) in enumerate(data_loader):
    print(f"Image {i}: {img.shape}, Label: {label}")
    if i == 5:
        break