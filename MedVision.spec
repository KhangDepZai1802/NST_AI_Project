# MedVision.spec
# Đặt file này ở thư mục GỐC của project (cùng cấp với src/)
# Chạy bằng: pyinstaller MedVision.spec --noconfirm

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Thu thập data files từ các thư viện cần thiết ────────────────────────────
ultralytics_datas  = collect_data_files('ultralytics')
torchvision_datas  = collect_data_files('torchvision')

# ── Hidden imports — PyInstaller hay bỏ sót ──────────────────────────────────
hidden = (
    collect_submodules('ultralytics')
    + collect_submodules('torchvision')
    + collect_submodules('torch')
    + collect_submodules('sklearn')
    + collect_submodules('cv2')
    + collect_submodules('PIL')
    + [
        'yaml', 'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'numpy', 'torch', 'torchvision', 'torchvision.models',
        'torchvision.transforms', 'ultralytics',
        'cv2', 'PIL', 'PIL.Image',
        'sklearn', 'sklearn.metrics',
        'scipy', 'scipy.special',
    ]
)

a = Analysis(
    ['src/main.py'],                  # Entry point
    pathex=['.'],
    binaries=[],
    datas=[
        # ── Tài nguyên của app ──────────────────────────────────────────────
        ('assets',        'assets'),          # logo, icons
        ('models',        'models'),          # AI models (.pt, .pth)
        ('config.yaml',   '.'),               # config ở thư mục gốc

        # ── Thư viện bên ngoài ──────────────────────────────────────────────
        *ultralytics_datas,
        *torchvision_datas,
    ],
    hiddenimports=hidden,
    hookspath=['.'],                  # hook-ultralytics.py nằm ở thư mục gốc
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Loại bỏ các thứ không cần → giảm kích thước file
        'matplotlib', 'notebook', 'IPython', 'ipykernel',
        'jupyter', 'sphinx', 'pytest', 'black', 'mypy',
        'tensorboard', 'tensorflow', 'keras',
        'pandas', 'seaborn', 'plotly',
        'tkinter', '_tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,            # dùng COLLECT (thư mục), không phải onefile
    name='MedVisionAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                         # nén binary để giảm kích thước
    console=False,                    # ẩn cửa sổ console (GUI app)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/logoNST.ico',        # icon cho file .exe
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MedVisionAI',               # tên thư mục output trong dist/
)
