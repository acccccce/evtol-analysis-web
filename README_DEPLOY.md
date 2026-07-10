# eVTOL 在线分析网页部署包

这是从本地 Streamlit 小程序整理出的独立部署版本。把本目录作为一个代码仓库上传到云平台，即可让别人通过网页使用，不依赖你本机路径。

## 目录内容

- `app.py`：在线网页主程序
- `flight_dynamics_report/evtol_flight_dynamics_analysis.py`：平飞配平与模态计算模型
- `data/eVTOL三视图与参数.docx`：默认参数数据
- `reports/`：可下载的 Word 技术报告
- `requirements.txt`：Python 依赖
- `.streamlit/config.toml`：云端运行配置
- `runtime.txt`：Streamlit Community Cloud 使用的 Python 版本
- `Dockerfile`：服务器、Render、Railway 等平台可用的容器部署文件

## 方案 A：Streamlit Community Cloud（最简单）

1. 新建一个 GitHub 仓库。
2. 把 `outputs/evtol_web_deploy/` 目录里的所有文件上传到仓库根目录。
3. 打开 Streamlit Community Cloud，选择该 GitHub 仓库。
4. Main file path 填：`app.py`。
5. 点击 Deploy。

部署完成后会得到一个公开网页链接，别人打开链接即可使用。

## 方案 B：Render / Railway / 服务器 Docker 部署

如果平台支持 Docker，直接使用本目录的 `Dockerfile`：

```bash
docker build -t evtol-analysis .
docker run -p 8501:8501 evtol-analysis
```

云平台通常会自动读取 Dockerfile。对外开放的端口为 `8501`。

## 本地预览

在本目录运行：

```powershell
python -m streamlit run app.py
```

或双击 `start_web_preview.bat`。本地浏览器请访问 `http://localhost:8502`，不要访问 `0.0.0.0:8502`；`0.0.0.0` 只是服务器监听地址。

## 注意事项

- 过渡段功率为工程估算值，不等同于电池输入功率。
- 默认数据和报告已内置；用户也可以在网页中上传同结构的 Word 数据文件重新计算。
- 如果要长期公开，建议给仓库写清楚项目用途和数据来源，避免别人误解为已完成适航级仿真工具。

