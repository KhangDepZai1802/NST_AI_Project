# src/core/chromosome_classifier.py
"""
Pipeline phân loại NST kết hợp 2 phương pháp:

  Phương pháp 1 — Kích thước mask (CHÍNH):
    YOLO mask → đo diện tích + tỷ lệ → xếp nhóm Denver A-G
    Không cần train thêm, hoạt động tốt trên ảnh thật.

  Phương pháp 2 — ResNet (HỖ TRỢ):
    Chỉ dùng để phát hiện NST X (chx) vì chX có hình dạng đặc trưng.
    Các lớp khác không dùng ResNet vì domain gap quá lớn.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter


# ── Ngưỡng phân nhóm Denver theo diện tích tương đối ─────────────────────────
# Chuẩn hoá: NST lớn nhất = 1.0
_DENVER_THRESHOLDS = [
    ("A", "NST 1–3 (lớn nhất)",      0.75),
    ("B", "NST 4–5",                  0.60),
    ("C", "NST 6–12, X",              0.40),
    ("D", "NST 13–15 (tâm đầu)",      0.28),
    ("E", "NST 16–18",                0.20),
    ("F", "NST 19–20",                0.13),
    ("G", "NST 21–22, Y (nhỏ nhất)",  0.0),
]

# Hội chứng dựa trên phân phối nhóm Denver
_SYNDROME_RULES = [
    (lambda g, t: g.get("G", 0) >= 5 and t == 47,
     "Trisomy 21 — Down syndrome hoặc 47,XYY",
     "Thừa 1 NST nhóm G"),

    (lambda g, t: g.get("D", 0) >= 7,
     "Trisomy 13 — Patau syndrome",
     "Thừa 1 NST nhóm D"),

    (lambda g, t: g.get("E", 0) >= 7,
     "Trisomy 18 — Edwards syndrome",
     "Thừa 1 NST nhóm E"),

    (lambda g, t: g.get("C", 0) <= 13 and t <= 45,
     "Monosomy X — Turner (45,X)",
     "Thiếu NST nhóm C, nghi mất NST X"),

    (lambda g, t: g.get("C", 0) >= 17 and t >= 47,
     "47,XXX hoặc Klinefelter (47,XXY)",
     "Thừa NST nhóm C, nghi thêm NST X"),
]


@dataclass
class ChromosomeFeature:
    index: int
    area_norm: float
    length: float
    width: float
    aspect_ratio: float
    denver_group: str
    resnet_label: str = ""


@dataclass
class AnalysisReport:
    total_count: int
    normal_count: int
    tolerance: int
    is_normal_count: bool
    sex_estimation: str
    sex_confidence: str
    syndrome_flags: List[str]
    risk_level: str
    group_sizes: dict
    size_stats: dict
    features: List[ChromosomeFeature] = field(default_factory=list)
    classifier_active: bool = False
    label_counts: dict = field(default_factory=dict)
    chromosome_labels: List[str] = field(default_factory=list)
    x_count: int = 0


class ChromosomeClassifier:

    NUM_CLASSES = 23
    CLASS_NAMES = sorted([f"ch{i}" for i in range(1, 23)] + ["chx"])

    def __init__(self, model_path: str = "models/best_CirNET_v2.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.resnet = None
        self._load_resnet(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_resnet(self, path: str):
        import os
        if not path or not os.path.isfile(path):
            print(f"⚠️ Không tìm thấy ResNet: {path}")
            return
        try:
            state   = torch.load(path, map_location=self.device, weights_only=False)
            fc_w    = state.get("fc.weight")
            if fc_w is None:
                raise ValueError("Không tìm thấy fc.weight")
            num_cls = fc_w.shape[0]
            net     = models.resnet18(weights=None)
            net.fc  = nn.Linear(net.fc.in_features, num_cls)
            net.load_state_dict(state, strict=True)
            self.resnet = net.to(self.device).eval()
            print(f"✅ ResNet loaded ({num_cls} lớp) — hỗ trợ detect chX")
        except Exception as e:
            print(f"⚠️ Lỗi load ResNet: {e}")
            self.resnet = None

    def reload(self, new_path: str):
        self.model_path = new_path
        self.resnet = None
        self._load_resnet(new_path)

    # ── Predict chX ───────────────────────────────────────────────────────────

    def _predict_chx(self, bgr_crop: np.ndarray) -> tuple:
        """Trả về (is_chx, confidence). Chỉ dùng cho NST nhóm C."""
        if self.resnet is None:
            return False, 0.0
        try:
            gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
            rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            t    = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.resnet(t), dim=1)[0]
                idx   = int(torch.argmax(probs).item())
                conf  = float(probs[idx].item())
            return self.CLASS_NAMES[idx] == "chx", conf
        except Exception:
            return False, 0.0

    # ── Trích xuất đặc trưng từ YOLO mask ────────────────────────────────────

    def _extract_features(self, yolo_result, orig_bgr) -> List[ChromosomeFeature]:
        features = []
        boxes = yolo_result.boxes
        masks = getattr(yolo_result, "masks", None)
        n     = len(boxes) if boxes is not None else 0
        if n == 0:
            return features

        h_img = w_img = 1
        if orig_bgr is not None:
            h_img, w_img = orig_bgr.shape[:2]

        raw_areas, raw_len, raw_wid = [], [], []

        for i in range(n):
            area = length = width = 0.0
            try:
                if masks is not None:
                    md   = masks.data[i].cpu().numpy()
                    mfull = cv2.resize(md.astype(np.float32), (w_img, h_img),
                                       interpolation=cv2.INTER_NEAREST)
                    mbin  = (mfull > 0.5).astype(np.uint8)
                    area  = float(np.sum(mbin))
                    cnts, _ = cv2.findContours(mbin, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        if len(cnt) >= 5:
                            e = cv2.fitEllipse(cnt)
                            length, width = max(e[1]), min(e[1]) + 1e-6
                        else:
                            x, y, bw, bh = cv2.boundingRect(cnt)
                            length, width = max(bw, bh), min(bw, bh) + 1e-6
                else:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    bw, bh = abs(x2 - x1), abs(y2 - y1)
                    area, length, width = bw * bh, max(bw, bh), min(bw, bh) + 1e-6
            except Exception:
                pass
            raw_areas.append(area)
            raw_len.append(length)
            raw_wid.append(width)

        max_area = max(raw_areas) or 1.0

        for i in range(n):
            area_norm    = raw_areas[i] / max_area
            ar           = raw_len[i] / raw_wid[i] if raw_wid[i] > 0 else 1.0
            denver_group = "G"
            for grp, _, thr in _DENVER_THRESHOLDS:
                if area_norm >= thr:
                    denver_group = grp
                    break
            features.append(ChromosomeFeature(
                index=i, area_norm=area_norm,
                length=raw_len[i], width=raw_wid[i],
                aspect_ratio=ar, denver_group=denver_group,
            ))
        return features

    # ── Detect NST X ──────────────────────────────────────────────────────────

    def _detect_x_count(self, features, yolo_result, orig_bgr) -> int:
        """Dùng ResNet để check NST nhóm C có phải chX không."""
        if orig_bgr is None or self.resnet is None:
            # Fallback: ước tính từ số lượng nhóm C
            c = sum(1 for f in features if f.denver_group == "C")
            return max(0, min(c - 14, 3))

        boxes = yolo_result.boxes
        masks = getattr(yolo_result, "masks", None)
        h, w  = orig_bgr.shape[:2]
        x_count = 0

        for f in features:
            if f.denver_group != "C":
                continue
            i = f.index
            try:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1 = max(0, x1 - 8), max(0, y1 - 8)
                x2, y2 = min(w, x2 + 8), min(h, y2 + 8)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = orig_bgr[y1:y2, x1:x2].copy()
                if masks is not None:
                    try:
                        md = masks.data[i].cpu().numpy()
                        mf = cv2.resize(md.astype(np.float32), (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                        mc = (mf[y1:y2, x1:x2] > 0.5).astype(np.uint8)
                        crop[mc == 0] = 255
                    except Exception:
                        pass
                is_x, conf = self._predict_chx(crop)
                if is_x and conf > 0.5:
                    x_count += 1
                    f.resnet_label = f"chx({conf:.0%})"
            except Exception:
                continue
        return x_count

    # ── Giới tính ─────────────────────────────────────────────────────────────

    def _estimate_sex(self, x_count: int, group_sizes: dict, has_resnet: bool):
        c = group_sizes.get("C", 0)
        g = group_sizes.get("G", 0)

        if has_resnet:
            if x_count >= 2:
                return "XX — Nữ (ước tính)", "Trung bình"
            elif x_count == 1:
                return ("XY — Nam (ước tính)", "Trung bình") if g >= 5 \
                    else ("X? — Cần xác nhận", "Thấp")
            else:
                return "Không xác định NST X", "Thấp"

        # Fallback không có ResNet
        if c >= 16:
            return "XX — Nữ (ước tính từ kích thước)", "Thấp"
        elif c == 15:
            return "XY — Nam (ước tính từ kích thước)", "Thấp"
        return f"Không xác định (nhóm C: {c})", "Thấp"

    # ── Hội chứng ─────────────────────────────────────────────────────────────

    def _detect_syndromes(self, group_sizes, total, normal, tolerance):
        flags = []
        for cond, name, desc in _SYNDROME_RULES:
            try:
                if cond(group_sizes, total):
                    flags.append(f"{name} ({desc})")
            except Exception:
                pass
        diff = abs(total - normal)
        if diff > tolerance and not flags:
            direction = "thừa" if total > normal else "thiếu"
            flags.append(f"Lệch bội {direction}: {total} NST ({direction} {diff})")
        return flags

    # ── PUBLIC: analyze ───────────────────────────────────────────────────────

    def analyze(self, yolo_result, normal=46, tolerance=1,
                orig_bgr=None) -> AnalysisReport:

        boxes     = yolo_result.boxes
        total     = len(boxes) if boxes is not None else 0
        is_normal = abs(total - normal) <= tolerance

        features    = self._extract_features(yolo_result, orig_bgr)
        group_sizes = {g: 0 for g in "ABCDEFG"}
        for f in features:
            group_sizes[f.denver_group] += 1

        x_count     = self._detect_x_count(features, yolo_result, orig_bgr)
        sex_est, sex_conf = self._estimate_sex(
            x_count, group_sizes, has_resnet=self.resnet is not None
        )
        syndrome_flags = self._detect_syndromes(group_sizes, total, normal, tolerance)

        if not is_normal and syndrome_flags:
            risk = "Nguy cơ cao"
        elif not is_normal or syndrome_flags:
            risk = "Cần theo dõi"
        else:
            risk = "Bình thường"

        areas = [f.area_norm for f in features]
        size_stats = {}
        if areas:
            size_stats = {
                "mean_area":  float(np.mean(areas)),
                "std_area":   float(np.std(areas)),
                "min_area":   float(np.min(areas)),
                "max_area":   float(np.max(areas)),
                "cv_percent": float(np.std(areas) / np.mean(areas) * 100)
                              if np.mean(areas) > 0 else 0,
            }

        labels      = [f.denver_group for f in features]
        label_counts = dict(Counter(labels))

        return AnalysisReport(
            total_count      = total,
            normal_count     = normal,
            tolerance        = tolerance,
            is_normal_count  = is_normal,
            sex_estimation   = sex_est,
            sex_confidence   = sex_conf,
            syndrome_flags   = syndrome_flags,
            risk_level       = risk,
            group_sizes      = group_sizes,
            size_stats       = size_stats,
            features         = features,
            classifier_active= True,
            label_counts     = label_counts,
            chromosome_labels= labels,
            x_count          = x_count,
        )