# 光催化反应降解率预测系统

本项目提供一个美观易用的 React GUI 前端和 Flask 后端，使用已训练好的 `encoders.pkl`、`scaler.pkl` 和 `xgb_best_model.pkl` 进行光催化反应降解率预测。无需在浏览器中加载 pickle，前端通过接口调用后端完成特征编码、缩放与模型推理，输出降解率百分比。

## 功能
- 条件输入：pH、催化剂质量(mg)、污染物浓度(mg/L)、反应时间(min)、A/B掺杂比例(0–100，支持小数)、污染物类型、掺杂元素（基体为 BiOBr）
- 结果展示：降解率百分比与可视化分析
- 数据处理：`LabelEncoder` 编码 + `StandardScaler` 缩放
- 模型预测：XGBoost 预训练模型（`xgb_best_model.pkl`）

## 目录结构
- `src/components/PredictionForm.tsx` 前端主界面与表单
- `src/services/modelService.ts` 前端服务，调用后端接口
- `server/predict_server.py` 后端 Flask 服务，加载 pkl 并预测
- `encoders.pkl` / `scaler.pkl` / `xgb_best_model.pkl` 放置在项目根目录

## 环境要求
- Node.js 18+
- Python 3.12+

## 快速开始
1. 安装前端依赖
   - `npm install`
2. 准备后端虚拟环境（安装在项目目录 D 盘路径下）
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
   - `pip install flask flask-cors numpy joblib scikit-learn xgboost`
3. 启动后端
   - `.\.venv\Scripts\python.exe server\predict_server.py`
   - 服务默认监听 `http://127.0.0.1:8000`
4. 启动前端
   - `npm run dev`
   - 访问 `http://localhost:5173/`

## 接口说明
- `POST /predict`
  - 请求体（JSON）：
    ```json
    {
      "pH": 7.0,
      "mass": 15,
      "concentration": 20,
      "time": 60,
      "aDopingRatio": 10.5,
      "bDopingRatio": 5.25,
      "pollutant": "RhB",
      "material": "B&Ce"
    }
    ```
  - 响应体：`{ "prediction": <百分比> }`

## 注意
- `encoders.pkl`、`scaler.pkl`、`xgb_best_model.pkl` 必须存在于项目根目录；如需自定义路径，请修改 `server/predict_server.py`。
- 由于 scikit-learn / xgboost 版本差异，加载旧版本序列化模型可能产生警告但可兼容运行；如需更稳妥，请在原版本中使用官方导出方法重新保存后加载。

## 许可
请在仓库中添加适合你的许可证（如 MIT）。

