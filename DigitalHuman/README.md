# DigitalMan - 数字人播报系统

基于 Flask 的数字人视频生成 Web 应用，支持上传头像、输入播报文本，通过 TTS 语音合成与 Wav2Lip 嘴型同步技术，自动生成带嘴型动画的数字人播报视频。

## 功能特性

- 🎙️ **多音色语音合成**：基于微软 Edge-TTS，支持 10 种中文音色（普通话、东北话、陕西话、粤语、台湾腔等）
- 🎬 **照片级嘴型同步**：GPU 环境下使用 Wav2Lip 实现高质量嘴型动画，CPU 环境下自动切换为快速模式
- 🖼️ **背景图支持**：可上传背景图片，系统自动抠图并将数字人自然融合到背景中
- ⚡ **异步任务处理**：后台线程生成视频，前端实时轮询进度
- 🎚️ **语速调节**：支持 0.5x ~ 2.0x 语速调节
- 🎨 **透明通道叠加**：支持 PNG 透明通道，人物边缘自然融入背景

## 技术栈

- **后端**：Flask >= 3.0, Flask-CORS
- **TTS**：edge-tts（微软 Edge 在线语音合成）
- **视频/图像**：OpenCV, Pillow, imageio, NumPy
- **深度学习**：PyTorch + Wav2Lip（嘴型同步）
- **前端**：原生 HTML5 / CSS3 / JavaScript

## 快速开始

### 环境要求

- Python >= 3.10
- NVIDIA GPU + CUDA（推荐，用于 Wav2Lip 嘴型同步；无 GPU 也可运行，自动降级为 CPU 模式）

### 1. 克隆仓库

```bash
git clone https://github.com/Jurf666/DigitalHuman.git
cd DigitalHuman
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. 安装依赖

> ⚠️ **注意**：`requirements.txt` 中的 `torch` 默认是 CPU 版本。如果你有 NVIDIA GPU，请务必安装 CUDA 版本的 PyTorch：

```bash
# 有 GPU（CUDA 12.x）
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 无 GPU（CPU 版本）
pip install -r requirements.txt
```

### 4. 下载 Wav2Lip 模型

项目依赖 Wav2Lip 预训练模型，模型文件较大（约 415MB），未包含在仓库中。请下载后放到指定目录：

- 下载地址：[wav2lip_gan.pth](https://huggingface.co/spaces/Rudrabha/Wav2Lip/resolve/main/wav2lip_gan.pth)
- 放置路径：`digital_human/wav2lip/models/wav2lip_gan.pth`

```bash
# 创建模型目录
mkdir -p digital_human/wav2lip/models

# 手动下载 wav2lip_gan.pth 放到上述目录
```

### 5. 启动服务

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:5000`

## 项目结构

```
DigitalMan/
├── app.py                          # Flask 应用入口
├── requirements.txt                # Python 依赖
├── LICENSE                         # MIT 开源协议
├── README.md                       # 项目说明
├── digital_human/                  # 核心功能包
│   ├── tts_engine.py              # TTS 语音合成（edge-tts）
│   ├── video_generator.py         # 数字人视频生成器
│   ├── wav2lip_engine.py          # Wav2Lip 推理封装
│   ├── utils.py                   # 工具函数
│   └── wav2lip/                   # Wav2Lip 源码与模型
│       └── models/
│           └── wav2lip_gan.pth    # 预训练模型（需手动下载）
├── templates/
│   └── index.html                 # 前端页面
└── static/
    ├── css/style.css              # 样式
    ├── js/main.js                 # 前端交互
    ├── uploads/                   # 用户上传文件（运行时生成）
    └── outputs/                   # 生成的音频/视频（运行时生成）
```

## 使用说明

1. 在网页上传数字人形象照片（支持 JPG / PNG，推荐正面清晰人脸）
2. （可选）上传背景图片
3. 选择音色与语速
4. 输入播报文本（最多 5000 字）
5. 点击生成，等待后台处理
6. 完成后在弹窗中预览并下载视频

## 注意事项

- **网络要求**：edge-tts 需要联网调用微软服务，离线无法使用
- **文件大小**：单文件上传限制 50MB
- **GPU 显存**：Wav2Lip 推理需要一定显存，建议预留 2GB 以上
- **安全提醒**：当前使用 Flask 开发服务器运行，仅适用于本地开发测试，**不要直接暴露在公网**

## 后续扩展方向

- 接入声音克隆模型（如 GPT-SoVITS、CosyVoice），实现音色参考音频功能
- 字幕叠加（ffmpeg drawtext）
- 全身动画（SadTalker / LivePortrait 等方案）
- 任务队列持久化（Redis / SQLite）

## License

[MIT](LICENSE)
