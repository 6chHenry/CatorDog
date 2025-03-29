import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import os

# 加载训练好的模型
# ResNet 变种
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResBlock(64, 128)
        self.res2 = ResBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

model = ResNet()
model.load_state_dict(torch.load("RESNETPLUS.pth"))  # 替换成你训练好的模型路径
model.eval()

# Flask设置
app = Flask(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 上传图片的路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # 加载图片并进行预处理
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加一个batch维度

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = 'cat' if predicted.item() == 0 else 'dog'

    return render_template('result.html', label=label, filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
