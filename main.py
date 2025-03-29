# 导入Flask应用实例
from app import app  # 假设Flask应用在app.py文件中

if __name__ == '__main__':
    # 启动Flask应用
    app.run(host='127.0.0.1', port=5000, debug=False)  # 在本地启动Flask服务器
