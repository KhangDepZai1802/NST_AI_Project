# hook-ultralytics.py — đặt ở thư mục gốc project
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
datas   = collect_data_files('ultralytics')
hiddenimports = collect_submodules('ultralytics')