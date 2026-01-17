# 🎨 SmartBoard - AI-Powered Finger Writing System

An intelligent drawing application that uses computer vision and hand tracking to let you draw in the air using just your fingers. Built with OpenCV and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🖐️ Gesture-Based Drawing** - Draw using your index finger
- **✋ Palm Erasing** - Open your palm to erase
- **✌️ Shape Mode** - Peace sign to draw shapes (lines, rectangles, circles, arrows)
- **🎨 8-Color Palette** - Quick color switching with keyboard shortcuts
- **↩️ Undo/Redo** - Full history support with up to 10 states
- **💾 Auto-Save** - Automatic canvas saving every 30 seconds
- **📊 Real-Time FPS** - Performance monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smartboard.git
   cd smartboard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 🖐️ Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ Index finger only | Drawing mode |
| 🖐️ Open palm (all fingers) | Eraser mode |
| ✌️ Peace sign (index + middle) | Shape drawing mode |
| ✊ Fist (no fingers) | Pause |

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-8` | Change colors (red, green, blue, yellow, purple, cyan, white, orange) |
| `-` / `+` | Decrease / Increase brush size |
| `Space` | Cycle through shapes |
| `Enter` | Complete shape (in shape mode) |
| `Z` | Undo |
| `X` | Redo |
| `C` | Clear canvas |
| `S` | Save drawing |
| `T` | Toggle trail effect |
| `Q` | Quit |

## 📁 Project Structure

```
smartboard/
├── main.py               # Application entry point
├── config.py             # Configuration and constants
├── camera_manager.py     # Camera handling with fallback
├── gesture_recognizer.py # Hand detection and gestures
├── canvas_manager.py     # Drawing and history management
├── ui_renderer.py        # UI elements rendering
├── file_manager.py       # File I/O and auto-save
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Configuration

Edit `config.py` to customize:

- Detection confidence thresholds
- Brush sizes and colors
- Auto-save interval
- Camera resolution preferences

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand tracking solution
- [OpenCV](https://opencv.org/) - Computer vision library
