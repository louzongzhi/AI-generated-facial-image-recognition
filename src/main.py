import os
import paddle
import numpy as np
import pandas as pd
from PIL import Image
from paddle.vision.models import resnet101


model = resnet101(pretrained=True)
model.set_dict(paddle.load('src/model.pdparams'))
model.eval()

test_data_path = './testdata'
test_data = []
test_labels = []
for filename in os.listdir(test_data_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(test_data_path, filename)
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img.astype('float32')
        img = img / 255.0
        test_data.append(img)
        test_labels.append(filename.split('_')[0])
test_data = np.array(test_data)
test_labels = np.array(test_labels)

test_data = paddle.to_tensor(test_data)
preds = model(test_data)
preds = paddle.argmax(preds, axis=1)

result = pd.DataFrame({'filename': test_labels, 'label': preds.numpy()})
result.to_csv('./cla_pre.csv', index=False, header=None)
print('预测结果已保存到cla_pre.csv')
