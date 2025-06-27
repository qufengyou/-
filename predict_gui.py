import sys
import torch
from torchvision import transforms
from PIL import Image
from model_cnn import create_model
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import numpy as np

class PredictWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Driving Behavior Prediction')
        self.setGeometry(100, 100, 800, 400)  # 减小窗口高度
        
        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # 左侧布局（图片显示区域）
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #cccccc; }")
        left_layout.addWidget(self.image_label)
        
        # 选择图片按钮
        self.select_button = QPushButton('Select Image')
        self.select_button.clicked.connect(self.selectImage)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        left_layout.addWidget(self.select_button)
        
        layout.addLayout(left_layout)
        
        # 右侧布局（预测结果显示区域）
        right_layout = QVBoxLayout()
        
        # 预测结果显示
        self.result_container = QWidget()
        result_layout = QVBoxLayout(self.result_container)
        
        # 状态标签
        self.status_label = QLabel()
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
                padding: 10px;
            }
        """)
        result_layout.addWidget(self.status_label)
        
        # 概率标签
        self.probability_label = QLabel()
        self.probability_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #4CAF50;
                padding: 5px;
            }
        """)
        result_layout.addWidget(self.probability_label)
        
        self.result_container.setVisible(False)  # 初始时隐藏
        right_layout.addWidget(self.result_container)
        right_layout.addStretch()
        
        layout.addLayout(right_layout)
        
        # 状态映射
        self.class_names = {
            0: "Safe Driving",
            1: "Texting - Right",
            2: "Talking on Phone - Right",
            3: "Texting - Left",
            4: "Talking on Phone - Left",
            5: "Operating Radio",
            6: "Drinking",
            7: "Reaching Behind",
            8: "Hair and Makeup",
            9: "Talking to Passenger"
        }
        
    def loadModel(self):
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        try:
            self.model = create_model(self.device)
            checkpoint = torch.load('best_model11.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 设置图像转换
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            sys.exit(1)
    
    def selectImage(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_name:
            # 显示图片
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # 预测图片
            self.predictImage(file_name)
    
    def predictImage(self, image_path):
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 获取最高概率的类别
                max_prob, predicted_class = torch.max(probabilities, 0)
                percentage = int(max_prob.item() * 100)
                
                # 更新状态显示
                self.status_label.setText(f"Status: {self.class_names[predicted_class.item()]}")
                self.probability_label.setText(f"Confidence: {percentage}%")
                
                # 显示结果
                self.result_container.setVisible(True)
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")

def main():
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle('Fusion')
    window = PredictWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
