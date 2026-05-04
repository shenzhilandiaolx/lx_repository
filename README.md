# Camera Calibration Component (Python, NumPy only)

不使用 OpenCV 的相机标定组件，包含：
- 棋盘格角点提取（Harris + NMS + 网格拟合）
- Zhang 标定法内参估计

## 安装

```bash
pip install numpy pillow
```

## CLI

```bash
python -m camera_calibration.cli ./images --cols 9 --rows 6 --square-size 25 --output calibration_result.npz
```

## 代码示例

```python
import numpy as np
from PIL import Image
from camera_calibration import CameraCalibrator, CheckerboardSpec

board = CheckerboardSpec(cols=9, rows=6, square_size=25.0)
calibrator = CameraCalibrator(board)

gray_images = [
    np.asarray(Image.open("images/1.jpg").convert("L"), dtype=np.float64),
    np.asarray(Image.open("images/2.jpg").convert("L"), dtype=np.float64),
    np.asarray(Image.open("images/3.jpg").convert("L"), dtype=np.float64),
]

result = calibrator.calibrate_from_images(gray_images)
print(result.camera_matrix)
print(result.rms)
```
