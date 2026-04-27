# src/core/resource_path.py
"""
Helper để lấy đường dẫn tài nguyên đúng cách,
dù chạy từ source code hay từ file .exe đã đóng gói.

Cách dùng:
    from src.core.resource_path import rp
    config_path = rp("config.yaml")
    model_path  = rp("models/best.pt")
    logo_path   = rp("assets/logoNST.png")
"""

import sys
import os


def rp(relative_path: str) -> str:
    """
    Trả về đường dẫn tuyệt đối đúng cho cả 2 trường hợp:
    - Chạy từ source code:   dùng thư mục gốc project
    - Chạy từ file .exe:     dùng sys._MEIPASS (thư mục tạm của PyInstaller)
    """
    if hasattr(sys, '_MEIPASS'):
        # Đang chạy từ .exe — PyInstaller giải nén tài nguyên vào _MEIPASS
        base = sys._MEIPASS
    else:
        # Đang chạy từ source — lấy thư mục gốc project
        # src/core/resource_path.py → lên 2 cấp → project root
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
    return os.path.join(base, relative_path)


def get_writable_dir(subdir: str = "") -> str:
    """
    Trả về thư mục có thể ghi được (để lưu báo cáo, export CSV, v.v.)
    Khi chạy .exe, _MEIPASS là read-only, nên cần dùng thư mục khác.

    - Chạy từ source: thư mục gốc project
    - Chạy từ .exe:   thư mục chứa file .exe (dist/MedVisionAI/)
    """
    if hasattr(sys, '_MEIPASS'):
        # Thư mục chứa file .exe thực thi
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
    path = os.path.join(base, subdir) if subdir else base
    os.makedirs(path, exist_ok=True)
    return path
