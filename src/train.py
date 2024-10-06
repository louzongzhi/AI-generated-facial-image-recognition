from data import RealFakeDataLoader


data_path = 'data'
cropSize = 256
batch_size = 32
num_threads = 4
validation_split = 0.2

data_loader = RealFakeDataLoader(data_path, cropSize, batch_size, num_threads, validation_split)
train_loader = data_loader.train_dataloader
val_loader = data_loader.val_dataloader