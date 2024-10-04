import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet101

class ResNet101(nn.Layer):
    def __init__(self, num_classes=2):
        super(ResNet101, self).__init__()
        self.backbone = resnet101(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

model = ResNet101(num_classes=2)
model = paddle.Model(model)

# model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
#               paddle.nn.CrossEntropyLoss(),
#               paddle.metric.Accuracy(topk=(1, 5)))
# model.fit(train_data, epochs=10, batch_size=32, verbose=1)
# model.evaluate(test_data, batch_size=32, verbose=1)
# model.save('src')

# model = paddle.load('src')
# model = paddle.Model(model)
# model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
#               paddle.nn.CrossEntropyLoss(),
#               paddle.metric.Accuracy(topk=(1, 5)))
# model.evaluate(test_data, batch_size=32, verbose=1)
# preds = model.predict(test_data, batch_size=32)
# preds = np.argmax(preds, axis=1)
# result = pd.DataFrame({'filename': test_labels, 'label': preds})
# result.to_csv('result.csv', index=False)