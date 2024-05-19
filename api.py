# 构建api
from flask import Flask, request, jsonify

from sklearn.linear_model import LinearRegression
import numpy as np

# 生成示例数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

app = Flask(__name__)

# 定义API端点
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # 获取POST请求中的JSON数据
    x_value = data['x']  # 提取输入特征值

    # 使用模型进行预测
    prediction = model.predict([[x_value]])

    # 返回预测结果
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

