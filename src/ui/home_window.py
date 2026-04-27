# src/ui/home_window.py
"""
MedVision AI — Trang chủ (Kraken Style)
Primary: #7132f5  |  Background: #ffffff  |  Text: #101114
"""

import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QGridLayout,
    QScrollArea, QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve  # [ADD] animation imports
from PyQt6.QtGui import QIcon, QColor, QPixmap
from src.core.resource_path import rp

# ══════════════════════════════════════════════════════════════════════════════
# SHARED — stylesheet dùng chung cả 2 màn hình
# ══════════════════════════════════════════════════════════════════════════════

KRAKEN_QSS = """
* { font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif; }
QMainWindow { background: #f7f7fb; }

/* ── BtnPrimary: mặc định viền tím nền trắng chữ tím | hover: nền tím chữ trắng ── */
#BtnPrimary {
    background: #ffffff; color: #7132f5; border: 1.5px solid #7132f5;
    border-radius: 10px; font-size: 14px; font-weight: bold;
    min-height: 44px; padding: 0 24px;
}
#BtnPrimary:hover   { background: #7132f5; color: #ffffff; border-color: #7132f5; }
#BtnPrimary:pressed { background: #5e22d4; color: #ffffff; border-color: #5e22d4; }
#BtnPrimary:disabled { background: #f3f0fb; color: #b0a0d8; border-color: #d4bbff; }

/* ── BtnSecondary: mặc định viền tím nền trắng chữ tím | hover: nền tím chữ trắng ── */
#BtnSecondary {
    background: #ffffff; color: #7132f5; border: 1.5px solid #7132f5;
    border-radius: 10px; font-size: 13px; font-weight: bold;
    min-height: 40px; padding: 0 20px;
}
#BtnSecondary:hover   { background: #7132f5; color: #ffffff; border-color: #7132f5; }
#BtnSecondary:pressed { background: #5e22d4; color: #ffffff; border-color: #5e22d4; }
#BtnSecondary:disabled { background: #f3f0fb; color: #b0a0d8; border-color: #d4bbff; }

/* ── BtnGhost: mặc định viền tím nhạt nền trắng chữ tím | hover: nền tím chữ trắng ── */
#BtnGhost {
    background: #ffffff; color: #7132f5;
    border: 1.5px solid #7132f5; border-radius: 10px;
    font-size: 13px; font-weight: bold; min-height: 40px; padding: 0 20px;
}
#BtnGhost:hover   { background: #7132f5; color: #ffffff; border-color: #7132f5; }
#BtnGhost:pressed { background: #5e22d4; color: #ffffff; border-color: #5e22d4; }
#BtnGhost:disabled { background: #f3f0fb; color: #c0b8d8; border-color: #d4bbff; }

/* ── BtnHome: mặc định viền tím nền trắng chữ tím | hover: nền tím chữ trắng ── */
#BtnHome {
    background: #ffffff; color: #7132f5;
    border: 1.5px solid #7132f5; border-radius: 8px;
    font-size: 12px; font-weight: bold;
    min-height: 34px; padding: 0 16px;
}
#BtnHome:hover   { background: #7132f5; color: #ffffff; border-color: #7132f5; }
#BtnHome:pressed { background: #5e22d4; color: #ffffff; }

/* ── BtnDanger: nền đỏ nhạt chữ đỏ ── */
#BtnDanger {
    background: #fff0f0; color: #dc2626;
    border: 1.5px solid #fca5a5; border-radius: 10px;
    font-size: 13px; font-weight: bold;
    min-height: 40px; padding: 0 20px;
}
#BtnDanger:hover { background: #dc2626; color: #ffffff; border-color: #dc2626; }

#TopBar {
    background: #ffffff;
    border-bottom: 1px solid #ebebf0;
}
#Footer {
    background: #ffffff;
    border-top: 1px solid #ebebf0;
}
#FooterText { font-size: 11px; color: #b0b0c0; }

#LabelSection {
    font-size: 11px; font-weight: bold;
    color: #9090a8; letter-spacing: 1.4px;
}
#LabelClock {
    font-size: 13px; color: #9090a8;
    font-family: 'Consolas', monospace;
    min-width: 190px;
}

#BadgeActive {
    font-size: 11px; font-weight: bold;
    color: #15803d; background: #dcfce7;
    border-radius: 20px; padding: 3px 12px; border: none;
}
#BadgeSoon {
    font-size: 11px; color: #9090a8;
    background: #f0f0f8; border-radius: 20px;
    padding: 3px 12px; border: none;
}

#StatCard {
    background: #ffffff; border-radius: 14px; border: 1px solid #d4bbff;
}
#StatNum   { font-size: 32px; font-weight: 900; }
#StatLabel { font-size: 12px; color: #9090a8; }

#HistoryCard {
    background: #ffffff; border-radius: 14px; border: 1px solid #d4bbff;
    min-height: 120px;
}
#HistoryEmpty { font-size: 13px; color: #c0c0d0; padding: 20px; }

/* ── Scrollbar ── */
QScrollBar:vertical {
    background: #f0f0f8; width: 5px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #d0d0e8; border-radius: 3px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #7132f5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Tabs ── */
QTabWidget::pane {
    border: none; background: #ffffff;
    border-radius: 0 0 14px 14px;
}
QTabBar::tab {
    padding: 10px 18px; font-size: 12px; font-weight: bold;
    color: #9090a8; background: #f7f7fb;
    border: none; border-radius: 8px 8px 0 0; margin-right: 2px;
}
QTabBar::tab:selected {
    color: #7132f5; background: #ffffff;
    border-bottom: 3px solid #7132f5;
}
QTabBar::tab:hover:!selected { background: #f3eeff; color: #7132f5; }
"""


def _shadow(widget, blur=18, dy=3, alpha=14):
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setXOffset(0)
    s.setYOffset(dy)
    s.setColor(QColor(113, 50, 245, alpha))
    widget.setGraphicsEffect(s)


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED BUTTON — hover chậm bằng QTimer interpolation
# ══════════════════════════════════════════════════════════════════════════════

class AnimatedButton(QPushButton):
    """
    Nút với hiệu ứng hover chậm (fade 200ms, 20 bước).
    Mặc định: viền tím, nền trắng, chữ tím.
    Hover:     viền tím, nền tím, chữ trắng.
    Disabled:  viền xám nhạt, nền xám nhạt, chữ xám — vẫn đọc được.
    """

    _C_BG_IDLE   = (255, 255, 255)
    _C_TEXT_IDLE = (113,  50, 245)
    _C_BG_HOV    = (113,  50, 245)
    _C_TEXT_HOV  = (255, 255, 255)
    _C_BORDER    = "#7132f5"
    _STEPS       = 4
    _INTERVAL_MS = 3

    def __init__(self, text="", parent=None, radius=10, font_size=13, min_h=42):
        super().__init__(text, parent)
        self._radius    = radius
        self._font_size = font_size
        self._min_h     = min_h
        self._progress  = 0.0
        self._direction = 0
        self._applying  = False      # guard chống đệ quy vô tận
        self._timer     = QTimer(self)
        self._timer.setInterval(self._INTERVAL_MS)
        self._timer.timeout.connect(self._step)
        self._apply()

    @staticmethod
    def _lerp(a, b, t):
        return int(a + (b - a) * t)

    def _step(self):
        self._progress += self._direction / self._STEPS
        self._progress  = max(0.0, min(1.0, self._progress))
        self._apply()
        if self._progress in (0.0, 1.0):
            self._timer.stop()

    def _apply(self):
        # Guard: setStyleSheet() có thể trigger changeEvent → _apply() lại → đệ quy
        if self._applying:
            return
        self._applying = True
        try:
            if not self.isEnabled():
                self.setStyleSheet(
                    f"QPushButton{{background:#f3f0fb;color:#b0a0d8;"
                    f"border:1.5px solid #d4bbff;border-radius:{self._radius}px;"
                    f"font-size:{self._font_size}px;font-weight:bold;"
                    f"min-height:{self._min_h}px;padding:0 20px;}}"
                )
                return
            t  = self._progress
            bg = (self._lerp(self._C_BG_IDLE[0], self._C_BG_HOV[0], t),
                  self._lerp(self._C_BG_IDLE[1], self._C_BG_HOV[1], t),
                  self._lerp(self._C_BG_IDLE[2], self._C_BG_HOV[2], t))
            tx = (self._lerp(self._C_TEXT_IDLE[0], self._C_TEXT_HOV[0], t),
                  self._lerp(self._C_TEXT_IDLE[1], self._C_TEXT_HOV[1], t),
                  self._lerp(self._C_TEXT_IDLE[2], self._C_TEXT_HOV[2], t))
            self.setStyleSheet(
                f"QPushButton{{background:rgb({bg[0]},{bg[1]},{bg[2]});"
                f"color:rgb({tx[0]},{tx[1]},{tx[2]});"
                f"border:1.5px solid {self._C_BORDER};border-radius:{self._radius}px;"
                f"font-size:{self._font_size}px;font-weight:bold;"
                f"min-height:{self._min_h}px;padding:0 20px;}}"
            )
        finally:
            self._applying = False

    def enterEvent(self, event):
        if self.isEnabled():
            self._direction = 1
            self._timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.isEnabled():
            self._direction = -1
            self._timer.start()
        super().leaveEvent(event)

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        super().changeEvent(event)
        # Chỉ xử lý khi enabled/disabled thay đổi, tránh react với mọi event khác
        if event.type() == QEvent.Type.EnabledChange:
            if not self.isEnabled():
                self._timer.stop()
                self._progress = 0.0
            self._apply()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE CARD
# ══════════════════════════════════════════════════════════════════════════════

class ModuleCard(QFrame):
    clicked = pyqtSignal()

    _THEMES = {
        "purple": ("🧬", "#f3eeff", "#7132f5"),
        "red":    ("🩸", "#fff0f0", "#dc2626"),
        "green":  ("🦠", "#f0fdf4", "#16a34a"),
        "blue":   ("🫁", "#eff6ff", "#2563eb"),
    }
    # [KEEP] Card frame borders unchanged
    _BASE  = "QFrame{background:#ffffff;border-radius:16px;border:1px solid #ebebf0;}"
    _HOVER = "QFrame{background:#faf8ff;border-radius:16px;border:2px solid #7132f5;}"
    _SOON  = "QFrame{background:#fafafa;border-radius:16px;border:1px solid #ebebf0;}"

    def __init__(self, title, desc, color="purple", status="active", parent=None):
        super().__init__(parent)
        self.status = status
        self._active_base = self._BASE if status == "active" else self._SOON
        self.setStyleSheet(self._active_base)
        self.setFixedHeight(186)
        _shadow(self)
        if status == "active":
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Hover animation — dùng QTimer step đơn giản, đúng chiều cả 2 card
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(16)           # ~60 fps
        self._hover_anim_step = 0                   # 0 = base, _hover_total_steps = hover
        self._hover_anim_dir  = 1                   # 1 = vào, -1 = ra
        self._hover_total_steps = 12                # 12 × 16ms ≈ 192ms
        self._hover_timer.timeout.connect(self._anim_step)

        icon_char, icon_bg, _ = self._THEMES.get(color, self._THEMES["purple"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(8)

        # Icon bubble — border already none, keep as-is
        bubble = QFrame()
        bubble.setFixedSize(46, 46)
        bubble.setStyleSheet(f"QFrame{{background:{icon_bg};border-radius:12px;border:none;}}")
        bl = QHBoxLayout(bubble)
        bl.setContentsMargins(0, 0, 0, 0)
        li = QLabel(icon_char)
        li.setAlignment(Qt.AlignmentFlag.AlignCenter)
        li.setStyleSheet("font-size:20px;background:transparent;border:none;")
        bl.addWidget(li)

        lbl_t = QLabel(title)
        lbl_t.setWordWrap(True)
        # [CHANGE] Added explicit border:none to prevent inherited border from QFrame stylesheet
        lbl_t.setStyleSheet("font-size:15px;font-weight:bold;color:#101114;background:transparent;border:none;")

        lbl_d = QLabel(desc)
        lbl_d.setWordWrap(True)
        # [CHANGE] Added explicit border:none to prevent inherited border from QFrame stylesheet
        lbl_d.setStyleSheet("font-size:12px;color:#7a7a9a;line-height:1.6;background:transparent;border:none;")

        badge = QLabel("● Hoạt động" if status == "active" else "○ Sắp ra mắt")
        badge.setObjectName("BadgeActive" if status == "active" else "BadgeSoon")
        # [ADD] Ensure badge label also has no unwanted border
        badge.setStyleSheet(badge.styleSheet() + "border:none;background:transparent;")

        layout.addWidget(bubble)
        layout.addWidget(lbl_t)
        layout.addWidget(lbl_d, 1)
        layout.addWidget(badge)

    def _anim_step(self):
        # Tiến step theo chiều hiện tại
        self._hover_anim_step = max(0, min(
            self._hover_anim_step + self._hover_anim_dir,
            self._hover_total_steps
        ))
        t     = self._hover_anim_step
        ratio = t / self._hover_total_steps

        if self.status == "active":
            r = int(0xeb + (0x71 - 0xeb) * ratio)
            g = int(0xeb + (0x32 - 0xeb) * ratio)
            b = int(0xf0 + (0xf5 - 0xf0) * ratio)
            border_color = f"#{r:02x}{g:02x}{b:02x}"
            border_width = 1 + ratio
            bg_r = int(0xff + (0xfa - 0xff) * ratio)
            bg_g = int(0xff + (0xf8 - 0xff) * ratio)
            bg_b = 0xff
            bg = f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"
            self.setStyleSheet(
                f"QFrame{{background:{bg};border-radius:16px;"
                f"border:{border_width:.1f}px solid {border_color};}}"
            )

        # Dừng khi chạm đầu hoặc cuối
        if t <= 0:
            self._hover_timer.stop()
            self.setStyleSheet(self._active_base)
        elif t >= self._hover_total_steps:
            self._hover_timer.stop()
            if self.status == "active":
                self.setStyleSheet(self._HOVER)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.status == "active":
            self.clicked.emit()
        super().mousePressEvent(e)

    def enterEvent(self, e):
        if self.status == "active":
            self._hover_anim_dir = 1
            # Luôn start lại timer (kể cả đang chạy) để đổi chiều ngay
            self._hover_timer.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        if self.status == "active":
            self._hover_anim_dir = -1
            self._hover_timer.start()
        super().leaveEvent(e)


# ══════════════════════════════════════════════════════════════════════════════
# HOME WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class HomeWindow(QMainWindow):
    open_chromosome  = pyqtSignal()
    open_blood_cancer = pyqtSignal()
    open_malaria = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedVision AI")
        self.setMinimumSize(1100, 700)
        self.setWindowIcon(QIcon(rp("assets/logoNST.png")))

        # ── Frameless + drop shadow cho toàn cửa sổ ──────────────────────────
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        win_shadow = QGraphicsDropShadowEffect(self)
        win_shadow.setBlurRadius(32)
        win_shadow.setXOffset(0)
        win_shadow.setYOffset(6)
        win_shadow.setColor(QColor(0, 0, 0, 80))

        # ── Drag state ────────────────────────────────────────────────────────
        self._drag_pos = None

        self._build_ui()

        # Áp shadow lên central widget sau khi build xong
        self.centralWidget().setGraphicsEffect(win_shadow)
        self.centralWidget().setStyleSheet(
            "background:#f7f7fb;border-radius:10px;"
        )

        self.setStyleSheet(KRAKEN_QSS)
        self.showMaximized()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    # ── Drag window ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if not self.isMaximized():
                self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background:#f7f7fb;")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        v.addWidget(self._topbar())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:#f7f7fb;border:none;}")

        inner = QWidget()
        inner.setStyleSheet("background:#f7f7fb;")
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(40, 32, 40, 32)
        iv.setSpacing(32)

        iv.addWidget(self._hero())
        iv.addWidget(self._modules())
        iv.addWidget(self._bottom_row())
        iv.addStretch()

        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        v.addWidget(self._footer())

    def _topbar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(62)
        h = QHBoxLayout(bar)
        h.setContentsMargins(36, 0, 36, 0)
        h.setSpacing(14)

        logo = QLabel()
        logo.setFixedSize(34, 34) # Bạn có thể đổi số này nếu muốn logo to/nhỏ hơn
        logo.setStyleSheet("background:transparent; border:none;")

        # Trỏ đường dẫn tới file ảnh của bạn
        pixmap = QPixmap(rp("assets/logoNST.png")) 

        # Scale ảnh cho vừa vặn với kích thước 34x34 và làm mịn ảnh
        logo.setPixmap(pixmap.scaled(
            logo.width(), 
            logo.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

        col = QVBoxLayout()
        col.setSpacing(1)
        n1 = QLabel("MedVision AI")
        n1.setStyleSheet("font-size:15px;font-weight:bold;color:#101114;background:transparent;")
        n2 = QLabel("Hệ thống phân tích hình ảnh y tế")
        n2.setStyleSheet("font-size:11px;color:#9090a8;background:transparent;")
        col.addWidget(n1)
        col.addWidget(n2)

        pill = QLabel("v2.0")
        pill.setStyleSheet(
            "font-size:11px;font-weight:bold;color:#7132f5;"
            "background:#f3eeff;border-radius:10px;"
            "padding:3px 12px;border:1px solid #d4bbff;"
        )

        self.lbl_clock = QLabel()
        self.lbl_clock.setObjectName("LabelClock")
        self.lbl_clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

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
        h.addWidget(pill)
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

    def _hero(self):
        hero = QFrame()
        hero.setStyleSheet(
            "QFrame{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #7132f5,stop:1 #a855f7);"
            "border-radius:20px;border:none;}"
        )
        hero.setFixedHeight(126)
        _shadow(hero, blur=28, dy=8, alpha=40)

        h = QHBoxLayout(hero)
        h.setContentsMargins(36, 0, 36, 0)

        left = QVBoxLayout()
        left.setSpacing(6)
        t = QLabel("Chào mừng đến MedVision AI")
        t.setStyleSheet("font-size:22px;font-weight:900;color:#ffffff;background:transparent;")
        s = QLabel("Nền tảng AI phân tích hình ảnh y tế  ·  Hỗ trợ chẩn đoán lâm sàng")
        s.setStyleSheet("font-size:13px;color:#e0d0ff;background:transparent;")
        left.addWidget(t)
        left.addWidget(s)

        big = QLabel("🧬")
        big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big.setStyleSheet("font-size:52px;background:transparent;")

        h.addLayout(left, 1)
        h.addWidget(big)
        return hero

    def _modules(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        lbl = QLabel("MODULES PHÂN TÍCH")
        lbl.setObjectName("LabelSection")
        v.addWidget(lbl)

        grid = QGridLayout()
        grid.setSpacing(16)

        defs = [
            ("Phân tích nhiễm sắc thể",
             "Phân đoạn, đếm NST · Phân nhóm Denver A–G · Phát hiện lệch bội và hội chứng di truyền",
             "purple", "active"),
            ("Phát hiện ung thư tế bào máu",
             "Chức năng Phân tích tế bào máu sử dụng trí tuệ nhân tạo (AI) để tự động nhận diện và phân loại các tế bào máu từ hình ảnh kính hiển vi.",
             "red", "active"),
            ("Phát hiện ký sinh trùng sốt rét",
             "Phát hiện Plasmodium trong hồng cầu qua tiêu bản máu nhuộm Giemsa",
             "green", "active"),
            ("Phát hiện tổn thương phổi",
             "Phân tích X-quang ngực · Viêm phổi · Tràn dịch · U phổi",
             "blue", "coming_soon"),
        ]

        signals = [self.open_chromosome, self.open_blood_cancer, self.open_malaria]
        signal_idx = 0
        for i, (title, desc, color, status) in enumerate(defs):
            card = ModuleCard(title, desc, color, status)
            if status == "active":
                card.clicked.connect(signals[signal_idx].emit)
                signal_idx += 1
            grid.addWidget(card, i // 2, i % 2)

        v.addLayout(grid)
        return w

    def _bottom_row(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(16)
        h.addWidget(self._stats_section(), 1)
        h.addWidget(self._history_section(), 1)
        return w

    def _stats_section(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        lbl = QLabel("PHIÊN LÀM VIỆC")
        lbl.setObjectName("LabelSection")
        v.addWidget(lbl)

        g = QGridLayout()
        g.setSpacing(12)
        self._st_total   = self._stat_card("0", "Ảnh đã phân tích",    "#7132f5")
        self._st_normal  = self._stat_card("0", "Kết quả bình thường", "#16a34a")
        self._st_warning = self._stat_card("0", "Cần theo dõi",        "#d97706")
        g.addWidget(self._st_total, 0, 0)
        g.addWidget(self._st_normal, 0, 1)
        g.addWidget(self._st_warning, 0, 2)
        v.addLayout(g)
        v.addStretch()
        return w

    def _stat_card(self, num, label, color):
        f = QFrame()
        f.setObjectName("StatCard")
        _shadow(f, blur=12, dy=2, alpha=10)
        v = QVBoxLayout(f)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(4)
        ln = QLabel(num)
        ln.setObjectName("StatNum")
        ln.setStyleSheet(f"color:{color};font-size:32px;font-weight:900;background:transparent;")
        ll = QLabel(label)
        ll.setObjectName("StatLabel")
        ll.setWordWrap(True)
        ll.setStyleSheet("background:transparent;")
        v.addWidget(ln)
        v.addWidget(ll)
        return f

    def _history_section(self):
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        lbl = QLabel("LỊCH SỬ GẦN ĐÂY")
        lbl.setObjectName("LabelSection")
        v.addWidget(lbl)

        self._hist_frame = QFrame()
        self._hist_frame.setObjectName("HistoryCard")
        _shadow(self._hist_frame, blur=12, dy=2, alpha=10)
        self._hist_layout = QVBoxLayout(self._hist_frame)
        self._hist_layout.setContentsMargins(16, 12, 16, 12)
        self._hist_layout.setSpacing(0)

        self._lbl_empty = QLabel("Chưa có lịch sử trong phiên này.")
        self._lbl_empty.setObjectName("HistoryEmpty")
        self._lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hist_layout.addWidget(self._lbl_empty)

        v.addWidget(self._hist_frame, 1)
        return w

    def _footer(self):
        f = QFrame()
        f.setObjectName("Footer")
        f.setFixedHeight(34)
        h = QHBoxLayout(f)
        h.setContentsMargins(36, 0, 36, 0)
        l = QLabel("© 2025 MedVision AI  ·  Đồ án môn học  ·  Hỗ trợ nghiên cứu & giáo dục")
        l.setObjectName("FooterText")
        r = QLabel("Kết quả AI không thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa")
        r.setObjectName("FooterText")
        r.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(l)
        h.addStretch()
        h.addWidget(r)
        return f

    # ── Public API ────────────────────────────────────────────────────────────

    def _tick(self):
        self.lbl_clock.setText(datetime.now().strftime("%d/%m/%Y   %H:%M:%S"))

    def update_session_stats(self, total: int, normal: int, warning: int):
        for card, val in [
            (self._st_total, total),
            (self._st_normal, normal),
            (self._st_warning, warning),
        ]:
            lbls = card.findChildren(QLabel)
            if lbls:
                lbls[0].setText(str(val))

    def add_history_item(self, filename: str, count: int, result: str, is_normal: bool):
        if self._lbl_empty.isVisible():
            self._lbl_empty.hide()

        row = QFrame()
        row.setFixedHeight(44)
        row.setStyleSheet(
            "QFrame{border:none;border-bottom:1px solid #f0f0f8;background:transparent;}"
        )
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        dot_color = "#16a34a" if is_normal else "#d97706"
        for widget, style, stretch in [
            (QLabel("●"),        f"color:{dot_color};font-size:9px;border:none;background:transparent;", False),
            (QLabel(filename),   "font-size:12px;color:#101114;border:none;background:transparent;", True),
            (QLabel(f"{count} NST"), "font-size:11px;color:#9090a8;border:none;background:transparent;", False),
            (QLabel(result),     f"font-size:11px;font-weight:bold;color:{'#15803d' if is_normal else '#b45309'};border:none;background:transparent;", False),
        ]:
            widget.setStyleSheet(style)
            if stretch:
                rl.addWidget(widget, 1)
            else:
                rl.addWidget(widget)

        self._hist_layout.insertWidget(0, row)
        while self._hist_layout.count() > 7:
            item = self._hist_layout.takeAt(self._hist_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()