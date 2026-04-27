# src/main.py
import sys
import os
import ctypes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from src.ui.home_window import HomeWindow
from src.ui.main_window import MainWindow   # ChromosomeWindow
from src.ui.blood_cancer_window import BloodCancerWindow
from src.ui.malaria_window import MalariaWindow


class AppController:
    """
    Điều phối toàn bộ app:
      - HomeWindow  : trang chủ chọn module
      - MainWindow  : module phân tích NST
      (Sau này thêm BloodCellWindow, MalariaWindow, LungWindow...)
    """

    def __init__(self, app: QApplication):
        self.app  = app
        self.home = HomeWindow()
        self.chrom_window: MainWindow = None
        self.blood_window: BloodCancerWindow = None
        self.malaria_window: MalariaWindow = None

        # Kết nối signal
        self.home.open_chromosome.connect(self._open_chromosome)
        self.home.open_blood_cancer.connect(self._open_blood_cancer)
        self.home.open_malaria.connect(self._open_malaria)
        self.home.show()
        

        # Thống kê session
        self._total   = 0
        self._normal  = 0
        self._warning = 0

    # ── Mở module NST ────────────────────────────────────────────────────────

    def _open_chromosome(self):
        if self.chrom_window is None:
            self.chrom_window = MainWindow(on_close_callback=self._back_to_home)
            # Hook vào on_analysis_finished để cập nhật stats + history
            self.chrom_window._analysis_done_hook = self._on_chromosome_done

        self.home.hide()
        self.chrom_window.showMaximized()

    # ── Mở module Ung thư tế bào máu ───────────────────────────────────────

    def _open_blood_cancer(self):
        if self.blood_window is None:
            self.blood_window = BloodCancerWindow(on_close_callback=self._back_to_home)
            # Hook để bảng thống kê trang chủ nhận kết quả từ module máu
            self.blood_window._analysis_done_hook = self._on_chromosome_done 
            
        self.home.hide()
        self.blood_window.showMaximized()

    # ── Mở module Sốt rét ─────────────────────────────────────────────────
    def _open_malaria(self):
       if self.malaria_window is None:
           self.malaria_window = MalariaWindow(on_close_callback=self._back_to_home)
           self.malaria_window._analysis_done_hook = self._on_chromosome_done
       
       self.home.hide()
       self.malaria_window.showMaximized()

    # ── Quay về trang chủ ───────────────────────────────────────────────────

    def _back_to_home(self):
        if self.chrom_window:
            self.chrom_window.hide()
        if self.blood_window:
            self.blood_window.hide()
        if self.malaria_window:
            self.malaria_window.hide()
        self.home.show()
        self.home.showMaximized()

    # ── Nhận kết quả phân tích NST ──────────────────────────────────────────

    def _on_chromosome_done(self, filename: str, count: int,
                            is_normal: bool, result_text: str):
        self._total += 1
        if is_normal:
            self._normal += 1
        else:
            self._warning += 1

        self.home.update_session_stats(self._total, self._normal, self._warning)
        self.home.add_history_item(filename, count, result_text, is_normal)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        myappid = "medvision.ai.platform.v2"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from src.core.resource_path import rp
    app.setWindowIcon(QIcon(rp("assets/logoNST.png")))

    controller = AppController(app)
    sys.exit(app.exec())