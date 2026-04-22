# src/ui/main_window.py
"""
MedVision AI — Module Phân tích Nhiễm sắc thể (Kraken Style)
Logic giữ nguyên, UI theo design system #7132f5
"""

import os
import yaml
import cv2
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QFileDialog, QMessageBox, QDialog,
    QLineEdit, QFormLayout, QSpinBox, QGraphicsDropShadowEffect,
    QSizePolicy, QScrollArea, QTabWidget, QProgressBar, QTextBrowser,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor, QAction

from src.core.ai_model import ChromosomeAnalyzer
from src.core.exporter import Exporter
from src.ui.home_window import KRAKEN_QSS, _shadow, AnimatedButton
from src.core.resource_path import rp, get_writable_dir
# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS DIALOG — Kraken style
# ══════════════════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt hệ thống")
        self.setFixedWidth(560)
        self.setStyleSheet("""
            QDialog { background: #ffffff; }
            QLabel  { font-size: 13px; color: #5a5a72; background: transparent; }
            QLineEdit, QSpinBox {
                padding: 9px 12px; border: 1.5px solid #e0e0ec;
                border-radius: 8px; background: #fafafa;
                font-size: 13px; color: #101114;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 1.5px solid #7132f5; background: #faf8ff;
            }
            QSpinBox::up-button, QSpinBox::down-button { width: 22px; }
            QPushButton {
                min-height: 38px; min-width: 110px;
                border-radius: 9px; font-weight: bold;
                font-size: 13px; border: none;
            }
        """)
        v = QVBoxLayout(self)
        v.setContentsMargins(32, 28, 32, 28)
        v.setSpacing(18)

        hdr = QLabel("Cấu hình hệ thống")
        hdr.setStyleSheet(
            "font-size:18px;font-weight:900;color:#101114;"
            "border-bottom:1.5px solid #ebebf0;padding-bottom:12px;"
        )
        v.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.model_path_input       = QLineEdit(current_config.get("model_path", "models/best.pt"))
        self.model_classifier_input = QLineEdit(current_config.get("model_classifier_path", "") or "")
        self.model_overlap_input    = QLineEdit(current_config.get("model_overlap_path", "") or "")
        self.model_anomaly_input    = QLineEdit(current_config.get("model_anomaly_path", "") or "")
        self.normal_count_input = QSpinBox()
        self.normal_count_input.setRange(0, 200)
        self.normal_count_input.setValue(current_config.get("normal_count", 46))
        self.tolerance_input = QSpinBox()
        self.tolerance_input.setRange(0, 50)
        self.tolerance_input.setValue(current_config.get("tolerance", 1))

        def slbl(text, color):
            l = QLabel(text)
            l.setStyleSheet(f"color:{color};font-size:10px;font-weight:bold;letter-spacing:1px;")
            return l

        form.addRow(slbl("── MODEL PHÂN ĐOẠN (YOLO) ──", "#7132f5"))
        form.addRow("YOLO phân đoạn / đếm NST:", self.model_path_input)
        form.addRow(slbl("── MODEL PHÂN LOẠI NST (ResNet) ──", "#16a34a"))
        form.addRow("ResNet phân loại 23 lớp NST:", self.model_classifier_input)
        form.addRow(slbl("── TUỲ CHỌN NÂNG CAO ──", "#9090a8"))
        form.addRow("Model tách NST chồng:", self.model_overlap_input)
        form.addRow("Model hỗ trợ bất thường:", self.model_anomaly_input)
        form.addRow("Số NST chuẩn (2n):", self.normal_count_input)
        form.addRow("Sai số cho phép (±):", self.tolerance_input)
        v.addLayout(form)

        row = QHBoxLayout()
        row.addStretch()
        btn_c = QPushButton("Hủy bỏ")
        btn_c.setStyleSheet(
            "QPushButton{background:#f0f0f8;color:#5a5a72;border:1.5px solid #e0e0ec;}"
            "QPushButton:hover{background:#e8e8f0;}"
        )
        btn_c.clicked.connect(self.reject)
        btn_s = QPushButton("Lưu cấu hình")
        btn_s.setStyleSheet(
            "QPushButton{background:#7132f5;color:white;}"
            "QPushButton:hover{background:#5e22d4;}"
        )
        btn_s.clicked.connect(self.accept)
        row.addWidget(btn_c)
        row.addWidget(btn_s)
        v.addLayout(row)

    def get_config(self):
        return {
            "model_path":            self.model_path_input.text().strip(),
            "model_classifier_path": self.model_classifier_input.text().strip(),
            "model_overlap_path":    self.model_overlap_input.text().strip(),
            "model_anomaly_path":    self.model_anomaly_input.text().strip(),
            "normal_count":          self.normal_count_input.value(),
            "tolerance":             self.tolerance_input.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, analyzer, path):
        super().__init__()
        self.analyzer = analyzer
        self.path = path

    def run(self):
        try:
            self.finished.emit(self.analyzer.analyze(self.path))
        except Exception as e:
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PANEL — Kraken style (HTML content giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

class DetailPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(False)
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content.setStyleSheet("""
            QTextBrowser { background:transparent;border:none;
                           font-size:13px;color:#101114;padding:4px; }
            QScrollBar:vertical { background:#f0f0f8;width:5px;border-radius:3px; }
            QScrollBar::handle:vertical { background:#d0d0e8;border-radius:3px;min-height:28px; }
            QScrollBar::handle:vertical:hover { background:#7132f5; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)
        self.content.setHtml(
            "<p style='color:#9090a8;padding:14px;font-size:13px;'>"
            "Chưa có dữ liệu.<br>Chạy AI để xem kết quả chi tiết.</p>"
        )
        layout.addWidget(self.content)

    def update_report(self, r, total: int):
        sex_color = {"Cao": "#15803d", "Trung bình": "#b45309", "Thấp": "#dc2626"}
        sc = sex_color.get(r.sex_confidence, "#9090a8")

        expected = {"A": 6, "B": 4, "C": 16, "D": 6, "E": 6, "F": 4, "G": 4}
        group_desc = {
            "A": "NST 1–3", "B": "NST 4–5", "C": "NST 6–12, X",
            "D": "NST 13–15", "E": "NST 16–18", "F": "NST 19–20", "G": "NST 21–22, Y",
        }
        rows = ""
        for g, n in r.group_sizes.items():
            exp = expected.get(g, 0)
            bar = min(int(n * 14), 110)
            ok  = (n == exp)
            bc  = "#7132f5" if ok else ("#dc2626" if abs(n - exp) > 1 else "#d97706")
            rows += f"""
            <tr>
              <td style='padding:4px 10px 4px 4px;font-weight:bold;color:#101114;font-size:13px;'>
                Nhóm {g}</td>
              <td style='padding:4px 6px;color:#9090a8;font-size:11px;'>{group_desc.get(g,'')}</td>
              <td style='padding:4px 8px;'>
                <div style='display:inline-block;width:{bar}px;height:8px;
                     background:{bc};border-radius:4px;'></div></td>
              <td style='padding:4px 4px;font-weight:bold;font-size:14px;color:{bc};'>{n}</td>
              <td style='padding:4px 4px;color:#c0c0d0;font-size:11px;'>/{exp}</td>
            </tr>"""

        st = r.size_stats
        stats_html = ""
        if st:
            stats_html = f"""
            <div style='background:#faf8ff;border-radius:10px;
                        padding:10px 14px;margin-top:10px;
                        border:1px solid #e8e0ff;'>
              <div style='font-weight:bold;color:#7132f5;font-size:12px;margin-bottom:6px;'>
                Thống kê kích thước mask</div>
              <table style='font-size:12px;width:100%;'>
                <tr><td style='color:#9090a8;'>Diện tích TB</td>
                    <td style='font-weight:bold;color:#101114;text-align:right;'>
                      {st.get("mean_area",0):.3f}</td></tr>
                <tr><td style='color:#9090a8;'>Độ lệch chuẩn</td>
                    <td style='font-weight:bold;color:#101114;text-align:right;'>
                      {st.get("std_area",0):.3f}</td></tr>
                <tr><td style='color:#9090a8;'>Min / Max</td>
                    <td style='font-weight:bold;color:#101114;text-align:right;'>
                      {st.get("min_area",0):.3f} / {st.get("max_area",0):.3f}</td></tr>
                <tr><td style='color:#9090a8;'>Hệ số biến thiên</td>
                    <td style='font-weight:bold;color:#101114;text-align:right;'>
                      {st.get("cv_percent",0):.1f}%</td></tr>
              </table>
            </div>"""

        if r.syndrome_flags:
            items = "".join(
                f"<li style='color:#dc2626;margin:4px 0;font-size:12px;'>{s}</li>"
                for s in r.syndrome_flags)
            synd = f"""
            <div style='background:#fff0f0;border-left:3px solid #dc2626;
                 border-radius:0 8px 8px 0;padding:10px 14px;margin-top:10px;'>
              <div style='font-weight:bold;color:#dc2626;font-size:12px;margin-bottom:4px;'>
                Hội chứng nghi ngờ</div>
              <ul style='margin:0 0 0 14px;padding:0;'>{items}</ul>
            </div>"""
        else:
            synd = """
            <div style='background:#f0fdf4;border-left:3px solid #16a34a;
                 border-radius:0 8px 8px 0;padding:10px 14px;margin-top:10px;'>
              <span style='color:#15803d;font-size:12px;font-weight:bold;'>
                Không phát hiện hội chứng đặc trưng</span>
            </div>"""

        self.content.setHtml(f"""
        <div style='font-family:Segoe UI,Arial;font-size:13px;line-height:1.65;padding:4px;'>
          <div style='background:#faf8ff;border-radius:10px;padding:10px 14px;
                      margin-bottom:10px;border:1px solid #e8e0ff;'>
            <div style='font-weight:bold;color:#7132f5;font-size:12px;margin-bottom:4px;'>
              Giới tính ước tính</div>
            <span style='font-size:14px;font-weight:bold;color:#101114;'>{r.sex_estimation}</span>
            <span style='color:{sc};font-size:11px;'> — Độ tin cậy: {r.sex_confidence}</span>
            <div style='color:#9090a8;font-size:11px;margin-top:4px;'>
              Dựa trên hình học mask. Cần karyotype để xác nhận.</div>
          </div>
          <div style='font-weight:bold;color:#101114;font-size:12px;margin-bottom:6px;'>
            Phân nhóm Denver (A–G)</div>
          <table style='width:100%;border-collapse:collapse;'>{rows}</table>
          {stats_html}
          {synd}
          <div style='margin-top:12px;padding:9px 12px;background:#f7f7fb;
               border-radius:8px;border:1px solid #ebebf0;'>
            <span style='color:#9090a8;font-size:11px;'>
              Phân nhóm theo kích thước mask tương đối.<br>
              Kết quả phục vụ hỗ trợ nghiên cứu / giáo dục.</span>
          </div>
        </div>""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW — Kraken style
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, on_close_callback=None):
        super().__init__()
        self.setWindowTitle("MedVision AI — Phân tích Nhiễm sắc thể")
        self.setGeometry(100, 100, 1440, 880)
        self.setWindowIcon(QIcon(rp("assets/logoNST.png")))

        # ── Frameless + drop shadow ───────────────────────────────────────────
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos = None

        self.on_close_callback    = on_close_callback
        self._analysis_done_hook  = None

        self.current_image_path   = None
        self.last_result_bgr      = None
        self.last_count           = None
        self.last_report_plain    = ""
        self.last_analysis_report = None

        self.dot_count = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._anim_tick)

        self.load_config()
        self.ai_analyzer = ChromosomeAnalyzer(self.config)

        self._build_ui()

        # Drop shadow cho toàn cửa sổ
        win_shadow = QGraphicsDropShadowEffect(self)
        win_shadow.setBlurRadius(32)
        win_shadow.setXOffset(0)
        win_shadow.setYOffset(6)
        win_shadow.setColor(QColor(0, 0, 0, 80))
        self.centralWidget().setGraphicsEffect(win_shadow)
        self.centralWidget().setStyleSheet("background:#f7f7fb;border-radius:10px;")

        self.setStyleSheet(KRAKEN_QSS + self._extra_qss())

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick)
        self._clock_timer.start(1000)
        self._tick()

    # ── Extra QSS specific to this window ────────────────────────────────────

    def _extra_qss(self):
        return """
            #ImagePane {
                background: #fafafa; border-radius: 12px;
                border: 1.5px dashed #c4b0f7;
            }
            #ImgLabel { color: #c0c0d0; font-size: 13px; background: transparent; }
            #ImgPaneTitle {
                font-size: 10px; font-weight: bold; color: #9090a8;
                letter-spacing: 1px; padding: 6px 0 2px 0; background: transparent;
            }
            #SideCard {
                background: #ffffff; border-radius: 16px; border: 1px solid #d4bbff;
            }
            #CountDisplay {
                background: #faf8ff; border-radius: 10px;
                border: 1px solid #e8e0ff; padding: 14px;
                font-size: 14px; font-weight: bold; color: #101114;
            }
            #StatusBadge {
                font-size: 11px; font-weight: bold; color: #9090a8;
                background: #f7f7fb; border: 1px solid #ebebf0;
                border-radius: 8px; padding: 5px 10px;
            }
            QProgressBar { border-radius: 3px; background: #ebebf0; }
            QProgressBar::chunk { background: #7132f5; border-radius: 3px; }
            #BottomBar { background: #ffffff; border-top: 1px solid #ebebf0; }
            #BarText { font-size: 11px; color: #b0b0c0; }
            #CtrlPanel {
                background: #ffffff; border-radius: 16px; border: 1px solid #d4bbff;
            }
            #CtrlTitle {
                font-size: 10px; font-weight: bold; color: #9090a8;
                letter-spacing: 1.4px; border-bottom: 1px solid #f0f0f8;
                padding-bottom: 10px;
            }
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
        logo.setStyleSheet("QFrame{background:#f3eeff;border-radius:12px;border:none;}")
        ll = QHBoxLayout(logo)
        ll.setContentsMargins(0, 0, 0, 0)
        lc = QLabel("🧬")
        lc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lc.setStyleSheet("font-size:20px;background:transparent;border:none;")
        ll.addWidget(lc)

        col = QVBoxLayout()
        col.setSpacing(1)
        n1 = QLabel("MedVision AI")
        n1.setStyleSheet("font-size:14px;font-weight:bold;color:#101114;background:transparent;")
        n2 = QLabel("Phân tích Nhiễm sắc thể")
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

    # ── Control panel ─────────────────────────────────────────────────────────

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
        v.addWidget(lbl)

        self.btn_load = AnimatedButton("📂  Tải ảnh lên", font_size=14, min_h=44)
        self.btn_load.clicked.connect(self.load_image)

        self.btn_run = AnimatedButton("🧠  Chạy AI phân tích", font_size=13, min_h=40)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_ai_analysis)

        self.lbl_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_status.setObjectName("StatusBadge")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

        self.btn_reset = AnimatedButton("🔄  Tải lại (Reset)", font_size=13, min_h=40)
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self.reset_all)

        self.btn_settings = AnimatedButton("⚙️  Cài đặt", font_size=13, min_h=40)
        self.btn_settings.clicked.connect(self.open_settings)

        v.addWidget(self.btn_load)
        v.addWidget(self.btn_run)
        v.addWidget(self.lbl_status)
        v.addWidget(self.progress_bar)
        v.addWidget(self.btn_reset)
        v.addStretch()
        v.addWidget(self.btn_settings)
        return panel

    # ── Image card ────────────────────────────────────────────────────────────

    def _image_card(self):
        card = QFrame()
        card.setObjectName("SideCard")
        _shadow(card)
        h = QHBoxLayout(card)
        h.setContentsMargins(14, 14, 14, 14)
        h.setSpacing(14)

        for attr, title, placeholder in [
            ("lbl_image_original", "ẢNH GỐC", "Ảnh gốc\n(Chưa tải)"),
            ("lbl_image_result",   "KẾT QUẢ AI PHÂN TÍCH", "Kết quả AI\nphân tích"),
        ]:
            pane = QFrame()
            pane.setObjectName("ImagePane")
            pv = QVBoxLayout(pane)
            pv.setContentsMargins(0, 0, 0, 0)
            pv.setSpacing(4)

            lbl_t = QLabel(title)
            lbl_t.setObjectName("ImgPaneTitle")
            lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_img = QLabel(placeholder)
            lbl_img.setObjectName("ImgLabel")
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            setattr(self, attr, lbl_img)

            pv.addWidget(lbl_t)
            pv.addWidget(lbl_img, 1)
            h.addWidget(pane)
        return card

    # ── Result panel (tabs) ───────────────────────────────────────────────────

    def _result_panel(self):
        panel = QFrame()
        panel.setObjectName("SideCard")
        panel.setFixedWidth(296)
        _shadow(panel)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1 — Tóm tắt
        summary = QWidget()
        sv = QVBoxLayout(summary)
        sv.setContentsMargins(16, 16, 16, 16)
        sv.setSpacing(10)

        sec_lbl = QLabel("KẾT QUẢ PHÂN TÍCH")
        sec_lbl.setStyleSheet(
            "font-size:10px;font-weight:bold;color:#9090a8;"
            "letter-spacing:1.4px;border-bottom:1px solid #f0f0f8;"
            "padding-bottom:8px;background:transparent;"
        )

        self.lbl_count = QLabel("Kết quả đếm:\n—")
        self.lbl_count.setObjectName("CountDisplay")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{background:#f0f0f8;width:5px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#d0d0e8;border-radius:3px;min-height:24px;}"
            "QScrollBar::handle:vertical:hover{background:#7132f5;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        self.lbl_report = QTextBrowser()
        self.lbl_report.setOpenExternalLinks(False)
        self.lbl_report.setFrameShape(QFrame.Shape.NoFrame)
        self.lbl_report.setStyleSheet(
            "QTextBrowser{background:transparent;border:none;"
            "font-size:13px;color:#101114;}"
        )
        self.lbl_report.setHtml(
            "<p style='color:#9090a8;font-size:13px;padding:4px;'>"
            "Báo cáo chi tiết sẽ xuất hiện sau khi chạy AI.</p>"
        )
        scroll.setWidget(self.lbl_report)

        self.btn_save = AnimatedButton("💾  Lưu ảnh & báo cáo", font_size=13, min_h=40)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_result_bundle)

        self.btn_export = AnimatedButton("📊  Xuất báo cáo CSV", font_size=13, min_h=40)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)

        sv.addWidget(sec_lbl)
        sv.addWidget(self.lbl_count)
        sv.addWidget(scroll, 1)
        sv.addWidget(self.btn_save)
        sv.addWidget(self.btn_export)

        # Tab 2 — Chi tiết
        self.detail_panel = DetailPanel()

        self.tabs.addTab(summary,            "Tóm tắt")
        self.tabs.addTab(self.detail_panel,  "Chi tiết")

        v.addWidget(self.tabs)
        return panel

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _bottom_bar(self):
        bar = QFrame()
        bar.setObjectName("BottomBar")
        bar.setFixedHeight(30)
        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 0, 24, 0)
        ll = QLabel("Upload → Phân tích → Xem kết quả → Lưu / Xuất")
        ll.setObjectName("BarText")
        lr = QLabel("© MedVision AI — Hỗ trợ nghiên cứu & giáo dục")
        lr.setObjectName("BarText")
        lr.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(ll)
        h.addStretch()
        h.addWidget(lr)
        return bar
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tick(self):
        self.lbl_clock.setText(datetime.now().strftime("%d/%m/%Y  |  %H:%M:%S"))

    def _go_home(self):
        if self.on_close_callback:
            self.on_close_callback()

    def _anim_tick(self):
        self.dot_count = (self.dot_count + 1) % 4
        self.lbl_status.setText(f"Đang phân tích{'.' * self.dot_count}")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#b45309;"
            "background:#fefce8;border:1px solid #fde68a;"
            "border-radius:8px;padding:5px 10px;"
        )

    def _cv_to_qpixmap(self, cv_img, w, h):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        hh, ww, ch = rgb.shape
        qi = QImage(rgb.data, ww, hh, ch * ww, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qi).scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def _show_msg(self, title, text, error=False):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Information)
        ac = "#dc2626" if error else "#7132f5"
        ah = "#b91c1c" if error else "#5e22d4"
        msg.setStyleSheet(f"""
            QMessageBox{{background:#ffffff;min-width:360px;}}
            QLabel{{color:#101114;font-size:13px;min-width:0;background:transparent;}}
            QPushButton{{background:{ac};color:white;border:none;
                border-radius:8px;padding:9px 24px;
                font-weight:bold;font-size:13px;
                min-width:80px;margin-top:10px;}}
            QPushButton:hover{{background:{ah};}}
        """)
        msg.exec()

    # ── Logic (giữ nguyên 100%) ───────────────────────────────────────────────

    def load_image(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh NST", "",
            "Ảnh (*.png *.jpg *.jpeg *.tif *.tiff);;Tất cả (*.*)")
        if not fn:
            return
        img_cv = cv2.imread(fn)
        if img_cv is None or img_cv.size == 0:
            self._show_msg("Ảnh không hợp lệ",
                "Không đọc được file. Hãy chọn .jpg, .png hoặc .tif/.tiff.", error=True)
            return
        self.current_image_path   = fn
        self.last_result_bgr      = None
        self.last_count           = None
        self.last_report_plain    = ""
        self.last_analysis_report = None

        px = QPixmap(fn).scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
        if px.isNull():
            self._show_msg("Ảnh không hợp lệ", "Qt không hiển thị được file này.", error=True)
            self.current_image_path = None
            return

        self.lbl_image_original.setPixmap(px)
        self.lbl_image_original.setStyleSheet("border:none;background:transparent;")
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI phân tích")
        self.lbl_image_result.setStyleSheet("")
        self.lbl_count.setText("Kết quả đếm:\n—")
        self.lbl_report.setHtml(
            "<p style='color:#9090a8;font-size:13px;'>Báo cáo sẽ xuất hiện sau khi chạy AI.</p>")
        self.detail_panel.content.setHtml(
            "<p style='color:#9090a8;padding:14px;'>Chưa có dữ liệu. Chạy AI để xem chi tiết.</p>")
        self.lbl_status.setText("Trạng thái: Sẵn sàng")
        self.lbl_status.setStyleSheet("")
        self.btn_run.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_save.setEnabled(False)

    def reset_all(self):
        self.current_image_path   = None
        self.last_result_bgr      = None
        self.last_count           = None
        self.last_report_plain    = ""
        self.last_analysis_report = None
        self.lbl_image_original.clear()
        self.lbl_image_original.setText("Ảnh gốc\n(Chưa tải)")
        self.lbl_image_original.setStyleSheet("")
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI\nphân tích")
        self.lbl_image_result.setStyleSheet("")
        self.lbl_count.setText("Kết quả đếm:\n—")
        self.lbl_report.setHtml(
            "<p style='color:#9090a8;font-size:13px;'>Báo cáo sẽ xuất hiện sau khi chạy AI.</p>")
        self.detail_panel.content.setHtml(
            "<p style='color:#9090a8;padding:14px;'>Chưa có dữ liệu. Chạy AI để xem chi tiết.</p>")
        self.lbl_status.setText("Trạng thái: Sẵn sàng")
        self.lbl_status.setStyleSheet("")
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self._show_msg("Thông báo", "Hệ thống đã được làm mới.")

    def run_ai_analysis(self):
        if not self.current_image_path:
            return
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.animation_timer.start(400)
        self.worker = WorkerThread(self.ai_analyzer, self.current_image_path)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.start()

    def on_analysis_finished(self, results):
        annotated_bgr, count_result, report_html, report_plain, analysis_report = results
        self.last_result_bgr      = annotated_bgr
        self.last_count           = count_result
        self.last_report_plain    = report_plain
        self.last_analysis_report = analysis_report

        self.animation_timer.stop()
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Phân tích hoàn tất ✓")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#15803d;"
            "background:#dcfce7;border:1px solid #86efac;"
            "border-radius:8px;padding:5px 10px;"
        )

        w = self.lbl_image_result.width()
        h = self.lbl_image_result.height()
        self.lbl_image_result.setPixmap(self._cv_to_qpixmap(annotated_bgr, w, h))
        self.lbl_image_result.setStyleSheet("border:none;background:transparent;")

        self.lbl_count.setTextFormat(Qt.TextFormat.RichText)
        cc = "#15803d" if analysis_report.is_normal_count else "#dc2626"
        self.lbl_count.setText(
            f"<div style='text-align:center;'>"
            f"<span style='font-size:34px;font-weight:900;color:{cc};'>{count_result}</span>"
            f"<span style='font-size:14px;color:#9090a8;'> NST</span></div>"
        )

        self.lbl_report.setHtml(report_html)
        self.detail_panel.update_report(analysis_report, count_result)

        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_save.setEnabled(True)

        if not analysis_report.is_normal_count or analysis_report.syndrome_flags:
            self.tabs.setCurrentIndex(1)

        if self._analysis_done_hook:
            fname   = os.path.basename(self.current_image_path or "")
            is_norm = analysis_report.is_normal_count
            self._analysis_done_hook(fname, count_result, is_norm,
                                     "Bình thường" if is_norm else "Cần theo dõi")

    def on_analysis_failed(self, message: str):
        self.animation_timer.stop()
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Phân tích thất bại ✗")
        self.lbl_status.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#dc2626;"
            "background:#fff0f0;border:1px solid #fca5a5;"
            "border-radius:8px;padding:5px 10px;"
        )
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self._show_msg("Lỗi phân tích", message, error=True)

    def export_csv(self):
        if self.last_count is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Lưu báo cáo CSV",
            f"Bao_cao_NST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV (*.csv)")
        if fp:
            try:
                Exporter.export_to_csv(fp, self.current_image_path,
                                       self.last_count, self.last_report_plain)
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo:\n{fp}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def save_result_bundle(self):
        if not self.current_image_path or self.last_result_bgr is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu")
        if not folder:
            return
        try:
            rp, _ = Exporter.save_result_bundle(
                folder, self.current_image_path, self.last_result_bgr,
                self.last_count, self.last_report_plain,
                int(self.config.get("normal_count", 46)),
                int(self.config.get("tolerance", 1)))
            QMessageBox.information(
                self, "Đã lưu",
                f"Lưu thành công:\n{folder}\nFile: {os.path.basename(rp)}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def load_config(self):
        self.config_file = "config.yaml"
        from src.core.resource_path import rp, get_writable_dir
        # Đọc config từ bundle (read-only)
        self.config_file_src = rp("config.yaml")
        # Ghi config ra thư mục writable
        self.config_file = os.path.join(get_writable_dir(), "config.yaml")
        # Lần đầu chạy: copy config gốc sang writable dir
        if not os.path.exists(self.config_file):
            import shutil
            shutil.copy2(self.config_file_src, self.config_file)
        defaults = {
            "model_path": "models/best.pt",
            "model_classifier_path": "models/best_CirNET_v2.pth",
            "model_overlap_path": "",
            "model_anomaly_path": "",
            "normal_count": 46,
            "tolerance": 1,
        }
        if not os.path.exists(self.config_file):
            self.config = dict(defaults)
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        else:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self.config = {**defaults, **loaded}

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.get_config()
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f)
            self.ai_analyzer.config = self.config
            self.ai_analyzer.load_model()
            QMessageBox.information(self, "Thành công", "Đã lưu cấu hình!")

    def show_about(self):
        self._show_msg(
            "Giới thiệu & bản quyền",
            "MedVision AI — Phần mềm phân tích NST tích hợp mô hình AI.\n\n"
            "Tính năng: Phân đoạn · Đếm NST · Phân nhóm Denver · "
            "Phát hiện hội chứng di truyền (rule-based).\n\n"
            "© Bản quyền thuộc tác giả. "
            "Kết quả chỉ mang tính hỗ trợ — chẩn đoán do bác sĩ chuyên khoa quyết định.")