# src/core/blood_cancer_classifier.py
"""
Pipeline phân loại ung thư tế bào máu (ALL — Acute Lymphoblastic Leukemia)

Model: ResNet-50 fine-tuned
Classes:
  - Benign  : Tế bào lành tính (Hematogones)
  - Early   : Giai đoạn sớm ALL
  - Pre     : Giai đoạn tiền ALL
  - Pro     : Giai đoạn ALL nặng

Tích hợp vào MedVision AI — NST_AI_Project
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── Thông tin lớp ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']

CLASS_INFO = {
    'Benign': {
        'vi': 'Lành tính (Hematogones)',
        'color': '#27AE60',
        'risk': 'Thấp',
        'desc': 'Tế bào lymphoid không phải ung thư, thường gặp ở tuỷ xương bình thường.',
        'recommendation': 'Không có dấu hiệu ung thư. Theo dõi định kỳ.'
    },
    'Early': {
        'vi': 'ALL Giai đoạn sớm',
        'color': '#E67E22',
        'risk': 'Trung bình',
        'desc': 'Phát hiện tế bào blast giai đoạn đầu của bệnh ALL.',
        'recommendation': 'Cần xét nghiệm bổ sung và tư vấn bác sĩ huyết học.'
    },
    'Pre': {
        'vi': 'ALL Tiền giai đoạn',
        'color': '#E74C3C',
        'risk': 'Cao',
        'desc': 'Tế bào blast giai đoạn Pre-ALL, có khả năng tiến triển nhanh.',
        'recommendation': 'Cần điều trị y tế khẩn. Liên hệ bác sĩ chuyên khoa ung thư.'
    },
    'Pro': {
        'vi': 'ALL Pro-lymphocytic',
        'color': '#8E44AD',
        'risk': 'Rất cao',
        'desc': 'Dạng nặng nhất của ALL, tế bào blast pro-lymphocytic.',
        'recommendation': 'Cần can thiệp y tế ngay lập tức.'
    },
}

# ── Risk mapping ──────────────────────────────────────────────────────────────
RISK_ORDER = {'Thấp': 0, 'Trung bình': 1, 'Cao': 2, 'Rất cao': 3}


@dataclass
class BloodCellPrediction:
    """Kết quả dự đoán cho 1 ảnh."""
    predicted_class: str
    confidence: float
    top3: List[Tuple[str, float]]
    risk_level: str
    description_vi: str
    recommendation: str
    is_cancer: bool
    color: str


@dataclass
class BloodCancerReport:
    """Báo cáo tổng hợp."""
    prediction: BloodCellPrediction
    model_loaded: bool
    model_path: str
    img_size: Tuple[int, int] = (224, 224)


class BloodCancerClassifier:
    """
    Classifier ung thư tế bào máu (ALL) — ResNet-50.
    Nhận đầu vào là 1 ảnh tiêu bản máu, trả về phân loại + xác suất.
    """

    NUM_CLASSES = 4
    CLASS_NAMES = CLASS_NAMES
    IMG_SIZE    = 224
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_path: str = "models/best_BloodCancerNET.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

    # ── Load model ─────────────────────────────────────────────────────────────

    def _build_resnet50(self, num_classes: int) -> nn.Module:
        net = models.resnet50(weights=None)
        in_features = net.fc.in_features  # 2048
        net.fc = nn.Sequential(
            nn.Dropout(0.4),            # fc.0
            nn.Linear(in_features, 512),# fc.1
            nn.ReLU(),                  # fc.2
            nn.BatchNorm1d(512),        # fc.3
            nn.Dropout(0.2),            # fc.4
            nn.Linear(512, num_classes) # fc.5
        )
        return net

    def _load_model(self, path: str):
        if not path:
            return
        # Thử path gốc, nếu không tìm thấy thì resolve qua rp() (cho PyInstaller .exe)
        from src.core.resource_path import rp
        resolved = path if os.path.isfile(path) else rp(path)
        if not os.path.isfile(resolved):
            print(f"⚠️ BloodCancerClassifier: Không tìm thấy model: {path}")
            return
        path = resolved
        try:
            raw = torch.load(path, map_location=self.device, weights_only=False)

            # ── Bước 1: Chuẩn hóa về state_dict ──────────────────────────────
            def _is_state_dict(d):
                """Kiểm tra dict có phải state_dict thực sự không (keys là str, values có Tensor)."""
                if not isinstance(d, dict):
                    return False
                # Keys phải là string
                if not all(isinstance(k, str) for k in d.keys()):
                    return False
                # Phải có ít nhất 1 Tensor value
                return any(isinstance(v, torch.Tensor) for v in d.values())

            # Trường hợp 1: torch.save(model, path) → raw là nn.Module
            if isinstance(raw, nn.Module):
                state = raw.state_dict()
            elif isinstance(raw, dict):
                # Tìm key nào chứa OrderedDict / dict có tensor weights
                state = None
                # Ưu tiên các key phổ biến theo thứ tự
                for candidate in ('model_state_dict', 'state_dict', 'model',
                                   'net', 'network', 'weights'):
                    if candidate in raw and _is_state_dict(raw[candidate]):
                        state = raw[candidate]
                        print(f"  [INFO] Dùng key '{candidate}' làm state_dict")
                        break
                # Fallback: lấy value đầu tiên là dict CÓ TENSOR (state_dict thực sự)
                if state is None:
                    for k, v in raw.items():
                        if _is_state_dict(v) and len(v) > 5:
                            state = v
                            print(f"  [INFO] Fallback: dùng key '{k}' làm state_dict")
                            break
                # Nếu raw chính là state_dict (keys là str, values có Tensor)
                if state is None and _is_state_dict(raw):
                    state = raw
                # Cuối cùng: dùng raw và hy vọng
                if state is None:
                    state = raw
            else:
                state = raw

            # ── Bước 2: Bỏ prefix 'module.' nếu model được train bằng DataParallel
            # Đảm bảo state là dict với string keys trước khi kiểm tra
            if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()) \
                    and any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}

            # ── Bước 3: Detect num_classes từ layer fc cuối ───────────────────
            # Tìm tất cả weight tensor 2D có liên quan đến fc
            fc_weight_keys = [
                k for k in state.keys()
                if 'weight' in k and len(state[k].shape) == 2
                and ('fc' in k or 'classifier' in k or 'head' in k)
            ]

            if not fc_weight_keys:
                # Fallback: lấy weight tensor 2D có shape[0] nhỏ nhất trong toàn model
                all_weight_keys = [
                    k for k in state.keys()
                    if 'weight' in k and len(state[k].shape) == 2
                ]
                if not all_weight_keys:
                    raise ValueError("Không tìm thấy weight tensor trong state_dict")
                fc_weight_keys = all_weight_keys

            # Layer cuối cùng = key có thứ tự lớn nhất (sort theo tên)
            # Và num_classes = shape[0] của layer đó
            fc_weight_keys_sorted = sorted(fc_weight_keys)
            last_fc_key = fc_weight_keys_sorted[-1]
            num_classes = state[last_fc_key].shape[0]

            # ── Bước 4: Build và load ─────────────────────────────────────────
            net = self._build_resnet50(num_classes)

            # Thử strict=True trước, fallback sang strict=False nếu cấu trúc lệch nhẹ
            try:
                net.load_state_dict(state, strict=True)
                load_mode = "strict"
            except RuntimeError as e_strict:
                missing    = [k for k in net.state_dict() if k not in state]
                unexpected = [k for k in state if k not in net.state_dict()]
                print(f"  ⚠ strict=True thất bại, thử non-strict...")
                print(f"  Missing keys  ({len(missing)}): {missing[:3]}")
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
                net.load_state_dict(state, strict=False)
                load_mode = "non-strict"

            self.model = net.to(self.device).eval()
            print(f"✅ BloodCancerClassifier: Load OK [{load_mode}] ({num_classes} lớp) "
                  f"từ {os.path.basename(path)} | fc_key={last_fc_key} "
                  f"shape={list(state[last_fc_key].shape)}")
        except Exception as e:
            print(f"⚠️ Lỗi load BloodCancerClassifier: {e}")
            import traceback; traceback.print_exc()
            self.model = None

    def reload(self, new_path: str):
        self.model_path = new_path
        self.model = None
        self._load_model(new_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(self, image_input) -> Optional[BloodCellPrediction]:
        """
        Phân loại ảnh tế bào máu.

        Args:
            image_input: PIL.Image, numpy BGR array, hoặc đường dẫn file (str)

        Returns:
            BloodCellPrediction hoặc None nếu lỗi
        """
        if self.model is None:
            return None
        try:
            pil_img = self._to_pil(image_input)
            tensor  = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw_out = self.model(tensor)

            # ── Chuẩn hóa output về Tensor 2D [1, num_classes] ───────────────
            # Một số model trả về OrderedDict (torchvision segmentation, detection)
            # hoặc dict với key 'out', 'logits', v.v.
            if isinstance(raw_out, (dict,)) or hasattr(raw_out, 'keys'):
                # Ưu tiên các key phổ biến
                for key in ('out', 'logits', 'output', 'pred', 'scores'):
                    if key in raw_out:
                        raw_out = raw_out[key]
                        break
                else:
                    # Lấy value đầu tiên là Tensor
                    for v in raw_out.values():
                        if isinstance(v, torch.Tensor):
                            raw_out = v
                            break
            # Đảm bảo là Tensor
            if not isinstance(raw_out, torch.Tensor):
                raise ValueError(f"Model output không phải Tensor: {type(raw_out)}")

            # Flatten nếu cần (output shape [1, C] hoặc [1, C, 1, 1])
            if raw_out.dim() > 2:
                raw_out = raw_out.view(raw_out.size(0), -1)

            probs = torch.softmax(raw_out, dim=1)[0]
            num_cls_actual = probs.shape[0]

            # Đồng bộ CLASS_NAMES với số class thực của model
            if num_cls_actual != len(self.CLASS_NAMES):
                # Mở rộng hoặc cắt bớt CLASS_NAMES cho khớp
                base = CLASS_NAMES  # module-level list
                if num_cls_actual <= len(base):
                    names = base[:num_cls_actual]
                else:
                    names = base + [f"Class{i}" for i in range(len(base), num_cls_actual)]
            else:
                names = self.CLASS_NAMES

            top3_probs, top3_idxs = torch.topk(probs, min(3, num_cls_actual))

            pred_idx  = int(top3_idxs[0].item())
            pred_conf = float(top3_probs[0].item())
            pred_cls  = names[pred_idx]

            top3 = [(names[int(i.item())], float(p.item()))
                    for i, p in zip(top3_idxs, top3_probs)]

            info = CLASS_INFO.get(pred_cls, {})
            return BloodCellPrediction(
                predicted_class = pred_cls,
                confidence      = pred_conf,
                top3            = top3,
                risk_level      = info.get('risk', 'Không xác định'),
                description_vi  = info.get('vi', pred_cls),
                recommendation  = info.get('recommendation', ''),
                is_cancer       = pred_cls != 'Benign',
                color           = info.get('color', '#7F8C8D'),
            )
        except Exception as e:
            import traceback
            print(f"⚠️ Lỗi predict BloodCancer: {e}")
            traceback.print_exc()
            return None

    def predict_from_path(self, image_path: str) -> BloodCancerReport:
        """Predict từ đường dẫn file, trả về BloodCancerReport đầy đủ."""
        pred = self.predict(image_path)
        return BloodCancerReport(
            prediction   = pred,
            model_loaded = self.is_ready,
            model_path   = self.model_path,
        )

    # ── Grad-CAM ───────────────────────────────────────────────────────────────

    def generate_gradcam(self, image_input, target_class: int = None) -> Optional[np.ndarray]:
        """
        Tạo Grad-CAM heatmap overlay lên ảnh gốc.

        Args:
            image_input : PIL.Image, numpy BGR, hoặc đường dẫn file
            target_class: index class muốn visualize (None = dùng class dự đoán)

        Returns:
            numpy array BGR (uint8) cùng kích thước ảnh gốc, hoặc None nếu lỗi
        """
        if self.model is None:
            return None
        try:
            # ── 1. Chuẩn bị ảnh ──────────────────────────────────────────────
            pil_img  = self._to_pil(image_input)
            orig_w, orig_h = pil_img.size          # kích thước gốc để resize về sau

            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            tensor.requires_grad_(False)

            # ── 2. Hook vào layer4 (feature map cuối ResNet) ──────────────────
            features   = {}
            gradients  = {}

            def fwd_hook(module, inp, out):
                features['map'] = out                  # [1, 2048, 7, 7]

            def bwd_hook(module, grad_in, grad_out):
                gradients['map'] = grad_out[0]         # [1, 2048, 7, 7]

            h_fwd = self.model.layer4.register_forward_hook(fwd_hook)
            h_bwd = self.model.layer4.register_full_backward_hook(bwd_hook)

            # ── 3. Forward + backward ─────────────────────────────────────────
            self.model.eval()
            output = self.model(tensor)               # [1, num_classes]

            # Xác định target class
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())

            # Zero grad rồi backward theo class mục tiêu
            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()

            h_fwd.remove()
            h_bwd.remove()

            # ── 4. Tính Grad-CAM ──────────────────────────────────────────────
            grads   = gradients['map'].detach().cpu()  # [1, C, H, W]
            fmaps   = features['map'].detach().cpu()   # [1, C, H, W]

            # Global Average Pooling trên gradient → trọng số mỗi channel
            weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

            # Weighted sum các feature map
            cam = (weights * fmaps).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            cam = torch.relu(cam)                              # chỉ giữ activation dương

            # Normalize về [0, 1]
            cam = cam.squeeze().numpy()                        # [H, W]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            # ── 5. Resize CAM về kích thước ảnh gốc ──────────────────────────
            cam_uint8 = (cam * 255).astype(np.uint8)
            cam_resized = cv2.resize(cam_uint8, (orig_w, orig_h),
                                     interpolation=cv2.INTER_LINEAR)

            # ── 6. Tạo heatmap màu và overlay lên ảnh gốc ────────────────────
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)  # BGR

            # Ảnh gốc sang BGR numpy
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            orig_bgr = cv2.resize(orig_bgr, (orig_w, orig_h))

            # Blend: 55% ảnh gốc + 45% heatmap
            overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

            # ── 7. Thêm colorbar chú thích ────────────────────────────────────
            bar_w   = max(20, orig_w // 20)
            bar_h   = orig_h
            bar_img = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
            for row in range(bar_h):
                val = int(255 * (1.0 - row / bar_h))
                color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                          cv2.COLORMAP_JET)[0, 0]
                bar_img[row, :] = color

            # Label "High" / "Low"
            cv2.putText(bar_img, 'Hi', (2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(bar_img, 'Lo', (2, bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            result = np.hstack([overlay, bar_img])
            return result                              # BGR numpy array

        except Exception as e:
            import traceback
            print(f"⚠️ Grad-CAM lỗi: {e}")
            traceback.print_exc()
            return None

    # ── Helper ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_input) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            # Giả sử BGR (OpenCV)
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        raise ValueError(f"Không hỗ trợ kiểu dữ liệu: {type(image_input)}")

    # ── Build report HTML ──────────────────────────────────────────────────────

    def build_html_report(self, report: BloodCancerReport,
                          image_path: str = "") -> Tuple[str, str]:
        """
        Trả về (html_string, plain_string) cho UI và CSV export.
        """
        if report.prediction is None:
            return (
                "<span style='color:#C0392B;'>❌ Không thể phân loại. Kiểm tra model.</span>",
                "Lỗi: Không thể phân loại ảnh."
            )

        p = report.prediction
        risk_colors = {
            'Thấp': '#27AE60', 'Trung bình': '#E67E22',
            'Cao': '#C0392B', 'Rất cao': '#8E44AD'
        }
        rc = risk_colors.get(p.risk_level, '#7F8C8D')

        # Top-3 bars
        bars_html = ""
        for cls_name, conf in p.top3:
            bar_w = int(conf * 120)
            cls_color = CLASS_INFO.get(cls_name, {}).get('color', '#3498DB')
            bars_html += f"""
            <div style='margin:3px 0;'>
              <span style='display:inline-block;width:70px;font-weight:bold;
                    color:{cls_color};'>{cls_name}</span>
              <span style='display:inline-block;width:{bar_w}px;height:12px;
                    background:{cls_color};border-radius:3px;vertical-align:middle;'></span>
              <span style='margin-left:6px;color:#2C3E50;'>{conf*100:.1f}%</span>
            </div>"""

        cancer_tag = (
            "<span style='color:#C0392B;font-weight:bold;'>🔴 Nghi ngờ ung thư (ALL)</span>"
            if p.is_cancer else
            "<span style='color:#27AE60;font-weight:bold;'>🟢 Lành tính</span>"
        )

        html = f"""
        <div style='font-family:Segoe UI,Arial;font-size:13px;line-height:1.7;'>

          <div style='background:{p.color}18;border-left:4px solid {p.color};
               border-radius:6px;padding:12px;margin-bottom:10px;'>
            <div style='font-size:22px;font-weight:900;color:{p.color};'>
              {p.description_vi}
            </div>
            <div style='font-size:14px;color:#2C3E50;'>
              Độ tin cậy: <b>{p.confidence*100:.1f}%</b> &nbsp;|&nbsp; {cancer_tag}
            </div>
          </div>

          <b>① Mức độ rủi ro:</b>
          <span style='color:{rc};font-weight:bold;font-size:15px;'> {p.risk_level}</span><br>

          <b>② Mô tả:</b>
          <span style='color:#2C3E50;'>{CLASS_INFO.get(p.predicted_class, {}).get('desc','')}</span><br>

          <b>③ Khuyến nghị:</b>
          <span style='color:#2C3E50;'>{p.recommendation}</span><br><br>

          <b>④ Phân phối xác suất Top-3:</b>
          <div style='background:#F8F9FA;border-radius:6px;padding:10px;margin-top:4px;'>
            {bars_html}
          </div>

          <div style='margin-top:10px;'>
            <span style='color:#95A5A6;font-size:11px;'>
              ⚠ Kết quả AI hỗ trợ — cần xét nghiệm lâm sàng và tư vấn bác sĩ huyết học để xác nhận.
            </span>
          </div>
        </div>
        """

        plain = (
            f"Kết quả: {p.description_vi} ({p.confidence*100:.1f}%)\n"
            f"Mức độ rủi ro: {p.risk_level}\n"
            f"{'Ung thư ALL nghi ngờ' if p.is_cancer else 'Lành tính'}\n"
            f"Khuyến nghị: {p.recommendation}\n"
            f"Lưu ý: Kết quả hỗ trợ, cần xét nghiệm lâm sàng."
        )

        return html, plain