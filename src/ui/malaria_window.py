# src/ui/malaria_window.py
"""
Module phát hiện ký sinh trùng sốt rét — MedVision AI
Kraken Style: giống blood_cancer_window.py
"""

import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QFileDialog, QMessageBox,
    QSizePolicy, QTextBrowser, QProgressBar, QTabWidget, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon

from src.core.malaria_classifier import MalariaClassifier, MalariaReport
from src.core.exporter import Exporter
from src.ui.home_window import KRAKEN_QSS, _shadow, AnimatedButton
from src.core.resource_path import rp

# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════

class MalariaWorker(QThread):
    finished = pyqtSignal(object, str, str, object)
    failed   = pyqtSignal(str)

    def __init__(self, classifier: MalariaClassifier, path: str):
        super().__init__()
        self.classifier = classifier
        self.path = path

    def run(self):
        try:
            report      = self.classifier.predict_from_path(self.path)
            html, plain = self.classifier.build_html_report(report, self.path)
            pred_idx    = None
            if report.prediction is not None:
                try:
                    pred_idx = MalariaClassifier.CLASS_NAMES.index(
                        report.prediction.predicted_class)
                except ValueError:
                    pred_idx = None
            gradcam = self.classifier.generate_gradcam(self.path, target_class=pred_idx)
            self.finished.emit(report, html, plain, gradcam)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MALARIA WINDOW — Kraken Style
# ══════════════════════════════════════════════════════════════════════════════

class MalariaWindow(QMainWindow):

    def __init__(self, on_close_callback=None,
                 model_path: str = "models/best_MalariaNET.pth"):
        super().__init__()
        self.setWindowTitle("MedVision AI — Phát hiện Ký sinh trùng Sốt rét")
        self.setGeometry(100, 100, 1440, 880)
        self.setWindowIcon(QIcon(rp("assets/logoNST.png")))

        # ── Frameless + drop shadow ───────────────────────────────────────────
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos = None

        self.on_close_callback   = on_close_callback
        self._analysis_done_hook = None
        self.current_image_path  = None
        self.last_report: MalariaReport = None
        self.last_plain  = ""
        self.dot_count   = 0

        self.classifier = MalariaClassifier(model_path=model_path)

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._anim_tick)

        self._build_ui()

        # Drop shadow cho toàn cửa sổ
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        from PyQt6.QtGui import QColor
        win_shadow = QGraphicsDropShadowEffect(self)
        win_shadow.setBlurRadius(32)
        win_shadow.setXOffset(0)
        win_shadow.setYOffset(6)
        win_shadow.setColor(QColor(0, 0, 0, 80))
        self.centralWidget().setGraphicsEffect(win_shadow)
        self.centralWidget().setStyleSheet("background:#f7f7fb;border-radius:10px;")

        self.setStyleSheet(KRAKEN_QSS + self._extra_qss())

        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(1000)
        self._tick()

    # ── Extra QSS ─────────────────────────────────────────────────────────────

    def _extra_qss(self):
        return """
            #ImagePane {
                background:#fafafa; border-radius:12px;
                border:1.5px dashed #c4b0f7;
            }
            #ImgLabel { color:#c0c0d0; font-size:13px; background:transparent; }
            #ImgPaneTitle {
                font-size:10px; font-weight:bold; color:#9090a8;
                letter-spacing:1px; padding:6px 0 2px 0; background:transparent;
            }
            #SideCard {
                background:#ffffff; border-radius:16px; border:1px solid #d4bbff;
            }
            #CtrlPanel {
                background:#ffffff; border-radius:16px; border:1px solid #d4bbff;
            }
            #CtrlTitle {
                font-size:10px; font-weight:bold; color:#9090a8;
                letter-spacing:1.4px; border-bottom:1px solid #f0f0f8;
                padding-bottom:10px; background:transparent;
            }
            #StatusBadge {
                font-size:11px; font-weight:bold; color:#9090a8;
                background:#f7f7fb; border:1px solid #ebebf0;
                border-radius:8px; padding:5px 10px;
            }
            #ModelStatus {
                font-size:11px; background:#f7f7fb; border-radius:8px;
                padding:8px; border:1px solid #ebebf0;
            }
            QProgressBar { border-radius:3px; background:#ebebf0; }
            QProgressBar::chunk { background:#16a34a; border-radius:3px; }
            #BottomBar { background:#ffffff; border-top:1px solid #ebebf0; }
            #BarText { font-size:11px; color:#b0b0c0; }
        """

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background:#f7f7fb;")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        v.addWidget(self._topbar())

        body = QWidget()
        body.setStyleSheet("background:#f7f7fb;")
        bh = QHBoxLayout(body)
        bh.setContentsMargins(20, 20, 20, 20)
        bh.setSpacing(16)
        bh.addWidget(self._ctrl_panel())
        bh.addWidget(self._image_card(), 1)
        bh.addWidget(self._result_panel())

        v.addWidget(body, 1)
        v.addWidget(self._bottom_bar())

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _topbar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(62)
        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 0, 24, 0)
        h.setSpacing(14)

        logo = QFrame()
        logo.setFixedSize(46, 46)
        logo.setStyleSheet("QFrame{background:#f0fdf4;border-radius:12px;border:none;}")
        ll = QHBoxLayout(logo)
        ll.setContentsMargins(0, 0, 0, 0)
        lc = QLabel("🦠")
        lc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lc.setStyleSheet("font-size:20px;background:transparent;border:none;")
        ll.addWidget(lc)

        col = QVBoxLayout()
        col.setSpacing(1)
        n1 = QLabel("MedVision AI")
        n1.setStyleSheet("font-size:14px;font-weight:bold;color:#101114;background:transparent;")
        n2 = QLabel("Phát hiện Ký sinh trùng Sốt rét (Plasmodium)")
        n2.setStyleSheet("font-size:11px;color:#9090a8;background:transparent;")
        col.addWidget(n1)
        col.addWidget(n2)

        self.btn_home = AnimatedButton("← Trang chủ", radius=8, font_size=12, min_h=34)
        self.btn_home.setFixedHeight(34)
        self.btn_home.clicked.connect(self._go_home)

        self.lbl_clock = QLabel()
        self.lbl_clock.setObjectName("LabelClock")
        self.lbl_clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_clock.setMinimumWidth(190)

        # ── Window control buttons ────────────────────────────────────────────
        _WC_BASE = (
            "QPushButton{background:transparent;border:none;"
            "border-radius:6px;font-size:14px;color:#9090a8;"
            "min-width:32px;max-width:32px;min-height:28px;max-height:28px;}"
            "QPushButton:hover{background:#f0f0f8;color:#101114;}"
        )
        _WC_CLOSE = (
            "QPushButton{background:transparent;border:none;"
            "border-radius:6px;font-size:14px;color:#9090a8;"
            "min-width:32px;max-width:32px;min-height:28px;max-height:28px;}"
            "QPushButton:hover{background:#dc2626;color:#ffffff;}"
        )
        btn_min = QPushButton("─")
        btn_min.setStyleSheet(_WC_BASE)
        btn_min.setToolTip("Thu nhỏ")
        btn_min.clicked.connect(self.showMinimized)

        self._btn_max = QPushButton("□")
        self._btn_max.setStyleSheet(_WC_BASE)
        self._btn_max.setToolTip("Phóng to / Thu hồi")
        self._btn_max.clicked.connect(self._toggle_maximize)

        btn_close = QPushButton("✕")
        btn_close.setStyleSheet(_WC_CLOSE)
        btn_close.setToolTip("Đóng")
        btn_close.clicked.connect(self.close)

        wc_frame = QFrame()
        wc_frame.setStyleSheet("QFrame{background:transparent;border:none;}")
        wc_layout = QHBoxLayout(wc_frame)
        wc_layout.setContentsMargins(6, 0, 0, 0)
        wc_layout.setSpacing(2)
        wc_layout.addWidget(btn_min)
        wc_layout.addWidget(self._btn_max)
        wc_layout.addWidget(btn_close)

        h.addWidget(logo)
        h.addLayout(col)
        h.addStretch()
        h.addWidget(self.btn_home)
        h.addWidget(self.lbl_clock)
        h.addWidget(wc_frame)
        return bar

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self._btn_max.setText("□")
            self._btn_max.setToolTip("Phóng to")
        else:
            self.showMaximized()
            self._btn_max.setText("❐")
            self._btn_max.setToolTip("Thu hồi")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if not self.isMaximized():
                self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def _ctrl_panel(self):
        panel = QFrame()
        panel.setObjectName("CtrlPanel")
        panel.setFixedWidth(216)
        _shadow(panel)
        v = QVBoxLayout(panel)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)

        lbl = QLabel("ĐIỀU KHIỂN")
        lbl.setObjectName("CtrlTitle")

        self.btn_load = AnimatedButton("📂  Tải ảnh lên", font_size=14, min_h=44)
        self.btn_load.clicked.connect(self._load_image)

        # Nút AI — xanh lá (malaria/sốt rét)
        self.btn_run = AnimatedButton("🔬  Phân tích AI", font_size=13, min_h=40)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_analysis)

        self.lbl_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_status.setObjectName("StatusBadge")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)

        self.btn_reset = AnimatedButton("🔄  Tải lại (Reset)", font_size=13, min_h=40)
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._reset)

        model_ok = self.classifier.is_ready
        model_lbl = QLabel(
            "Model sẵn sàng" if model_ok
            else "Chưa có model\nĐặt file vào:\nmodels/best_MalariaNET.pth"
        )
        model_lbl.setObjectName("ModelStatus")
        model_lbl.setWordWrap(True)
        model_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_lbl.setStyleSheet(
            f"font-size:11px;"
            f"color:{'#15803d' if model_ok else '#dc2626'};"
            "background:#f7f7fb;border-radius:8px;padding:8px;border:1px solid #ebebf0;"
        )

        v.addWidget(lbl)
        v.addWidget(self.btn_load)
        v.addWidget(self.btn_run)
        v.addWidget(self.lbl_status)
        v.addWidget(self.progress)
        v.addWidget(self.btn_reset)
        v.addStretch()
        v.addWidget(model_lbl)
        return panel

    # ── Image card ────────────────────────────────────────────────────────────

    def _image_card(self):
        card = QFrame()
        card.setObjectName("SideCard")
        _shadow(card)
        h = QHBoxLayout(card)
        h.setContentsMargins(14, 14, 14, 14)
        h.setSpacing(14)

        # Pane ảnh gốc
        pane_orig = QFrame()
        pane_orig.setObjectName("ImagePane")
        pv1 = QVBoxLayout(pane_orig)
        pv1.setContentsMargins(0, 0, 0, 0)
        pv1.setSpacing(4)
        lbl_t1 = QLabel("ẢNH GỐC")
        lbl_t1.setObjectName("ImgPaneTitle")
        lbl_t1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig = QLabel("Ảnh tiêu bản máu\n(Chưa tải)")
        self.lbl_orig.setObjectName("ImgLabel")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        pv1.addWidget(lbl_t1)
        pv1.addWidget(self.lbl_orig, 1)

        # Pane Grad-CAM
        pane_cam = QFrame()
        pane_cam.setObjectName("ImagePane")
        pv2 = QVBoxLayout(pane_cam)
        pv2.setContentsMargins(0, 0, 0, 0)
        pv2.setSpacing(4)
        lbl_t2 = QLabel("GRAD-CAM  (vùng AI tập trung)")
        lbl_t2.setObjectName("ImgPaneTitle")
        lbl_t2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cam = QLabel("Chạy phân tích AI\nđể xem Grad-CAM")
        self.lbl_cam.setObjectName("ImgLabel")
        self.lbl_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cam.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        pv2.addWidget(lbl_t2)
        pv2.addWidget(self.lbl_cam, 1)

        h.addWidget(pane_orig, 1)
        h.addWidget(pane_cam, 1)
        return card

    # ── Result panel ──────────────────────────────────────────────────────────

    def _result_panel(self):
        panel = QFrame()
        panel.setObjectName("SideCard")
        panel.setFixedWidth(310)
        _shadow(panel)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1 — Kết quả
        tab1 = QWidget()
        t1v  = QVBoxLayout(tab1)
        t1v.setContentsMargins(16, 16, 16, 16)
        t1v.setSpacing(10)

        sec = QLabel("KẾT QUẢ PHÂN TÍCH")
        sec.setStyleSheet(
            "font-size:10px;font-weight:bold;color:#9090a8;"
            "letter-spacing:1.4px;border-bottom:1px solid #f0f0f8;"
            "padding-bottom:8px;background:transparent;"
        )

        self.lbl_main_result = QLabel("—")
        self.lbl_main_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_main_result.setWordWrap(True)
        self.lbl_main_result.setStyleSheet(
            "font-size:15px;font-weight:bold;color:#9090a8;"
            "background:#faf8ff;border-radius:12px;"
            "border:1px solid #e8e0ff;padding:14px;"
        )
        self.lbl_main_result.setMinimumHeight(70)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{background:#f0f0f8;width:5px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#d0d0e8;border-radius:3px;min-height:24px;}"
            "QScrollBar::handle:vertical:hover{background:#16a34a;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        self.lbl_result = QTextBrowser()
        self.lbl_result.setOpenExternalLinks(False)
        self.lbl_result.setFrameShape(QFrame.Shape.NoFrame)
        self.lbl_result.setStyleSheet(
            "QTextBrowser{background:transparent;border:none;font-size:13px;color:#101114;}"
        )
        self.lbl_result.setHtml(
            "<p style='color:#9090a8;font-size:13px;padding:4px;'>"
            "Tải ảnh và nhấn 'Phân tích AI' để xem kết quả.</p>"
        )
        scroll.setWidget(self.lbl_result)

        self.btn_save = AnimatedButton("💾  Lưu báo cáo", font_size=13, min_h=40)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_report)

        t1v.addWidget(sec)
        t1v.addWidget(self.lbl_main_result)
        t1v.addWidget(scroll, 1)
        t1v.addWidget(self.btn_save)

        # Tab 2 — Hướng dẫn
        tab2 = QWidget()
        t2v  = QVBoxLayout(tab2)
        t2v.setContentsMargins(16, 16, 16, 16)
        guide = QTextBrowser()
        guide.setFrameShape(QFrame.Shape.NoFrame)
        guide.setStyleSheet(
            "QTextBrowser{background:transparent;border:none;font-size:13px;color:#101114;}"
        )
        guide.setHtml("""
        <div style='font-family:Segoe UI,Arial;font-size:13px;line-height:1.7;'>
          <div style='font-weight:bold;color:#16a34a;font-size:15px;margin-bottom:12px;'>
            Hướng dẫn sử dụng</div>

          <b>1. Loại ảnh phù hợp</b><br>
          <span style='color:#5a5a72;'>Ảnh tiêu bản máu mỏng nhuộm Giemsa dưới kính hiển vi
          (JPG/PNG/TIFF). Ảnh cần rõ nét, zoom đủ để nhìn thấy từng tế bào.</span><br><br>

          <b>2. Quy trình</b><br>
          <span style='color:#5a5a72;'>Tải ảnh → Phân tích AI → Xem kết quả + Grad-CAM</span><br><br>

          <b>3. Kết quả phân loại:</b><br>
          <div style='background:#fff0f0;border-radius:8px;padding:10px;margin:4px 0;'>
            <b style='color:#dc2626;'>⚠ Parasitized</b><br>
            <span style='color:#5a5a72;font-size:12px;'>
              Phát hiện ký sinh trùng Plasmodium trong hồng cầu.
              Dấu hiệu nghi ngờ bệnh sốt rét.</span>
          </div>
          <div style='background:#f0fdf4;border-radius:8px;padding:10px;margin:4px 0;'>
            <b style='color:#16a34a;'>✓ Uninfected</b><br>
            <span style='color:#5a5a72;font-size:12px;'>
              Không phát hiện ký sinh trùng. Tế bào bình thường.</span>
          </div>
          <br>

          <b>4. Grad-CAM</b><br>
          <span style='color:#5a5a72;'>Bản đồ nhiệt cho thấy vùng AI tập trung phân tích
          — màu đỏ/vàng = vùng ảnh hưởng nhiều nhất đến kết quả.</span><br><br>

          <div style='background:#fefce8;border-left:3px solid #d97706;
               border-radius:0 8px 8px 0;padding:10px;'>
            <b style='color:#b45309;'>Lưu ý quan trọng:</b><br>
            <span style='color:#5a5a72;font-size:12px;'>
              Kết quả AI chỉ hỗ trợ sàng lọc ban đầu. Cần xét nghiệm
              lâm sàng (giọt đặc, PCR) để chẩn đoán chính xác bệnh sốt rét.</span>
          </div>
        </div>
        """)
        t2v.addWidget(guide)

        self.tabs.addTab(tab1, "Kết quả")
        self.tabs.addTab(tab2, "Hướng dẫn")
        v.addWidget(self.tabs)
        return panel

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _bottom_bar(self):
        bar = QFrame()
        bar.setObjectName("BottomBar")
        bar.setFixedHeight(30)
        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 0, 24, 0)
        ll = QLabel("Tải ảnh → Phân tích AI → Xem kết quả → Lưu báo cáo")
        ll.setObjectName("BarText")
        lr = QLabel("© MedVision AI — Hỗ trợ nghiên cứu & giáo dục")
        lr.setObjectName("BarText")
        lr.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(ll)
        h.addStretch()
        h.addWidget(lr)
        return bar

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _tick(self):
        self.lbl_clock.setText(datetime.now().strftime("%d/%m/%Y  |  %H:%M:%S"))

    def _anim_tick(self):
        self.dot_count = (self.dot_count + 1) % 4
        self.lbl_status.setText(f"Đang phân tích{'.' * self.dot_count}")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#b45309;"
            "background:#fefce8;border:1px solid #fde68a;"
            "border-radius:8px;padding:5px 10px;"
        )

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh tiêu bản máu", "",
            "Ảnh (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Tất cả (*.*)"
        )
        if not path:
            return
        self.current_image_path = path
        self.last_report = None
        self.last_plain  = ""

        px = QPixmap(path)
        if px.isNull():
            QMessageBox.warning(self, "Lỗi", "Không đọc được ảnh!")
            return

        w = self.lbl_orig.width() or 600
        h = self.lbl_orig.height() or 600
        self.lbl_orig.setPixmap(
            px.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)
        )
        self.lbl_orig.setStyleSheet("border:none;background:transparent;")
        self.btn_run.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.lbl_main_result.setText("—")
        self.lbl_main_result.setStyleSheet(
            "font-size:15px;font-weight:bold;color:#9090a8;"
            "background:#faf8ff;border-radius:12px;"
            "border:1px solid #e8e0ff;padding:14px;"
        )
        self.lbl_result.setHtml(
            "<p style='color:#9090a8;font-size:13px;'>Nhấn 'Phân tích AI' để chạy.</p>"
        )
        self.lbl_status.setText("Ảnh đã tải")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#15803d;"
            "background:#dcfce7;border:1px solid #86efac;"
            "border-radius:8px;padding:5px 10px;"
        )

    def _run_analysis(self):
        if not self.current_image_path:
            return
        if not self.classifier.is_ready:
            QMessageBox.critical(
                self, "Lỗi",
                "Model chưa load!\n\n"
                "Train model trên Colab và đặt file\n"
                "best_MalariaNET.pth vào thư mục models/"
            )
            return
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setVisible(True)
        self.anim_timer.start(400)

        self.worker = MalariaWorker(self.classifier, self.current_image_path)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, report, html, plain, gradcam):
        self.last_report = report
        self.last_plain  = plain
        self.anim_timer.stop()
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_result.setHtml(html)
        self.tabs.setCurrentIndex(0)

        # Grad-CAM
        if gradcam is not None:
            px = self._bgr_to_pixmap(gradcam)
            w  = self.lbl_cam.width()  or 600
            h  = self.lbl_cam.height() or 500
            self.lbl_cam.setPixmap(
                px.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            )
            self.lbl_cam.setStyleSheet("border:none;background:transparent;")
        else:
            self.lbl_cam.setText("Grad-CAM không khả dụng")

        p = report.prediction
        if p:
            is_inf = p.is_infected
            rc     = "#dc2626" if is_inf else "#15803d"
            bg     = "#fff0f0" if is_inf else "#f0fdf4"
            brd    = "#fca5a5" if is_inf else "#86efac"
            self.lbl_main_result.setText(
                f"{'⚠ ' if is_inf else '✓ '}{p.description_vi}\n"
                f"Độ tin cậy: {p.confidence*100:.1f}%"
            )
            self.lbl_main_result.setStyleSheet(
                f"font-size:14px;font-weight:bold;color:{rc};"
                f"background:{bg};border-radius:12px;"
                f"border:1.5px solid {brd};padding:14px;"
            )
            self.lbl_status.setText(
                f"{'Nhiễm sốt rét' if is_inf else 'Không nhiễm'} — {p.confidence*100:.1f}%"
            )
            self.lbl_status.setStyleSheet(
                f"font-size:11px;font-weight:bold;color:{rc};"
                f"background:{bg};border:1px solid {brd};"
                "border-radius:8px;padding:5px 10px;"
            )
            if self._analysis_done_hook:
                fname = os.path.basename(self.current_image_path or "")
                tag   = f"{'Sốt rét (+)' if is_inf else 'Âm tính'} ({p.predicted_class})"
                self._analysis_done_hook(fname, 0, not is_inf, tag)

    def _on_failed(self, msg):
        self.anim_timer.stop()
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.lbl_status.setText("Phân tích thất bại ✗")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#dc2626;"
            "background:#fff0f0;border:1px solid #fca5a5;"
            "border-radius:8px;padding:5px 10px;"
        )
        QMessageBox.critical(self, "Lỗi phân tích", msg)

    def _reset(self):
        self.current_image_path = None
        self.last_report = None
        self.last_plain  = ""
        self.lbl_orig.clear()
        self.lbl_orig.setText("Ảnh tiêu bản máu\n(Chưa tải)")
        self.lbl_orig.setStyleSheet("")
        self.lbl_cam.clear()
        self.lbl_cam.setText("Chạy phân tích AI\nđể xem Grad-CAM")
        self.lbl_cam.setStyleSheet("")
        self.lbl_main_result.setText("—")
        self.lbl_main_result.setStyleSheet(
            "font-size:15px;font-weight:bold;color:#9090a8;"
            "background:#faf8ff;border-radius:12px;"
            "border:1px solid #e8e0ff;padding:14px;"
        )
        self.lbl_result.setHtml(
            "<p style='color:#9090a8;font-size:13px;'>Chưa có kết quả. Tải ảnh và chạy AI.</p>"
        )
        self.lbl_status.setText("Trạng thái: Sẵn sàng")
        self.lbl_status.setStyleSheet("")
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_reset.setEnabled(False)

    def _save_report(self):
        if not self.last_plain:
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu")
        if not folder:
            return
        try:
            Exporter.save_result_bundle(
                folder_path       = folder,
                source_image_path = self.current_image_path,
                result_bgr_image  = None,
                count_value       = 0,
                status_plain      = self.last_plain,
                normal_count      = 0,
                tolerance         = 0,
            )
            QMessageBox.information(self, "Đã lưu", f"Báo cáo lưu tại:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def _go_home(self):
        if self.on_close_callback:
            self.on_close_callback()

    @staticmethod
    def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)