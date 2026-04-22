# src/core/malaria_classifier.py
"""
Pipeline phát hiện ký sinh trùng sốt rét (Plasmodium) trong tế bào máu.

Model: ResNet-18 fine-tuned trên NIH Malaria Dataset
Classes:
  - Parasitized : Tế bào bị nhiễm Plasmodium
  - Uninfected  : Tế bào bình thường

NIH Malaria Dataset: 27,558 ảnh, 2 lớp, 100×100px
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ── Thông tin lớp ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Parasitized', 'Uninfected']

CLASS_INFO = {
    'Parasitized': {
        'vi':    'Nhiễm ký sinh trùng (Plasmodium)',
        'en':    'Parasitized',
        'color': '#dc2626',
        'risk':  'Cao',
        'icon':  '⚠',
        'desc':  'Phát hiện ký sinh trùng Plasmodium trong tế bào hồng cầu. '
                 'Đây là dấu hiệu của bệnh sốt rét.',
        'recommendation': (
            'Cần xét nghiệm lâm sàng xác nhận (giọt đặc, PCR). '
            'Liên hệ cơ sở y tế ngay để điều trị kịp thời.'
        ),
    },
    'Uninfected': {
        'vi':    'Không nhiễm ký sinh trùng',
        'en':    'Uninfected',
        'color': '#16a34a',
        'risk':  'Thấp',
        'icon':  '✓',
        'desc':  'Không phát hiện ký sinh trùng trong tế bào hồng cầu.',
        'recommendation': (
            'Tế bào bình thường. Nếu có triệu chứng lâm sàng, '
            'nên xét nghiệm thêm để loại trừ.'
        ),
    },
}


@dataclass
class MalariaPrediction:
    predicted_class: str
    confidence: float
    top2: List[Tuple[str, float]]
    risk_level: str
    description_vi: str
    recommendation: str
    is_infected: bool
    color: str
    icon: str


@dataclass
class MalariaReport:
    prediction: Optional[MalariaPrediction]
    model_loaded: bool
    model_path: str


class MalariaClassifier:
    """
    Classifier phát hiện sốt rét — ResNet-18.
    Input: ảnh tế bào máu nhuộm Giemsa (JPG/PNG).
    Output: Parasitized / Uninfected + confidence.
    """

    CLASS_NAMES = CLASS_NAMES
    IMG_SIZE    = 224
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_path: str = "models/best_MalariaNET.pth"):
        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model      = None
        self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

    # ── Load model ────────────────────────────────────────────────────────────

    def _build_resnet18(self, num_classes: int) -> nn.Module:
        net = models.resnet18(weights=None)
        in_f = net.fc.in_features  # 512
        net.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        return net

    def _load_model(self, path: str):
        if not path or not os.path.isfile(path):
            print(f"⚠️ MalariaClassifier: Không tìm thấy model: {path}")
            return
        try:
            raw = torch.load(path, map_location=self.device, weights_only=False)

            # Chuẩn hoá về state_dict
            def _is_sd(d):
                return (isinstance(d, dict)
                        and all(isinstance(k, str) for k in d.keys())
                        and any(isinstance(v, torch.Tensor) for v in d.values()))

            if isinstance(raw, nn.Module):
                state = raw.state_dict()
            elif isinstance(raw, dict):
                state = None
                for key in ('model_state_dict', 'state_dict', 'model', 'net'):
                    if key in raw and _is_sd(raw[key]):
                        state = raw[key]; break
                if state is None and _is_sd(raw):
                    state = raw
                if state is None:
                    state = raw
            else:
                state = raw

            # Bỏ prefix 'module.' nếu DataParallel
            if isinstance(state, dict) and any(
                    k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}

            # Detect num_classes
            fc_keys = [k for k in state.keys()
                       if 'weight' in k and len(state[k].shape) == 2
                       and ('fc' in k or 'classifier' in k or 'head' in k)]
            if not fc_keys:
                fc_keys = [k for k in state.keys()
                           if 'weight' in k and len(state[k].shape) == 2]
            num_classes = state[sorted(fc_keys)[-1]].shape[0]

            net = self._build_resnet18(num_classes)
            try:
                net.load_state_dict(state, strict=True)
            except RuntimeError:
                net.load_state_dict(state, strict=False)

            self.model = net.to(self.device).eval()
            print(f"✅ MalariaClassifier: Load OK ({num_classes} lớp) "
                  f"từ {os.path.basename(path)}")
        except Exception as e:
            import traceback
            print(f"⚠️ Lỗi load MalariaClassifier: {e}")
            traceback.print_exc()
            self.model = None

    def reload(self, new_path: str):
        self.model_path = new_path
        self.model = None
        self._load_model(new_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image_input) -> Optional[MalariaPrediction]:
        if self.model is None:
            return None
        try:
            pil    = self._to_pil(image_input)
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out   = self.model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                top2_p, top2_i = torch.topk(probs, min(2, len(self.CLASS_NAMES)))

            pred_idx  = int(top2_i[0].item())
            pred_conf = float(top2_p[0].item())
            pred_cls  = self.CLASS_NAMES[pred_idx]
            top2 = [(self.CLASS_NAMES[int(i.item())], float(p.item()))
                    for i, p in zip(top2_i, top2_p)]

            info = CLASS_INFO.get(pred_cls, {})
            return MalariaPrediction(
                predicted_class = pred_cls,
                confidence      = pred_conf,
                top2            = top2,
                risk_level      = info.get('risk', '?'),
                description_vi  = info.get('vi', pred_cls),
                recommendation  = info.get('recommendation', ''),
                is_infected     = pred_cls == 'Parasitized',
                color           = info.get('color', '#7F8C8D'),
                icon            = info.get('icon', '?'),
            )
        except Exception as e:
            import traceback
            print(f"⚠️ Malaria predict lỗi: {e}")
            traceback.print_exc()
            return None

    def predict_from_path(self, image_path: str) -> MalariaReport:
        pred = self.predict(image_path)
        return MalariaReport(
            prediction   = pred,
            model_loaded = self.is_ready,
            model_path   = self.model_path,
        )

    # ── Grad-CAM ──────────────────────────────────────────────────────────────

    def generate_gradcam(self, image_input,
                         target_class: int = None) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            pil    = self._to_pil(image_input)
            orig_w, orig_h = pil.size
            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            features, grads = {}, {}
            h_f = self.model.layer4.register_forward_hook(
                lambda m, i, o: features.update({'map': o}))
            h_b = self.model.layer4.register_full_backward_hook(
                lambda m, gi, go: grads.update({'map': go[0]}))

            self.model.eval()
            out = self.model(tensor)
            if target_class is None:
                target_class = int(out.argmax(1).item())
            self.model.zero_grad()
            out[0, target_class].backward()
            h_f.remove(); h_b.remove()

            w   = grads['map'].detach().cpu().mean(dim=[2, 3], keepdim=True)
            cam = (w * features['map'].detach().cpu()).sum(1, keepdim=True)
            cam = torch.relu(cam).squeeze().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cam_r = cv2.resize((cam * 255).astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
            heat  = cv2.applyColorMap(cam_r, cv2.COLORMAP_JET)
            orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            orig_bgr = cv2.resize(orig_bgr, (orig_w, orig_h))
            overlay  = cv2.addWeighted(orig_bgr, 0.55, heat, 0.45, 0)
            return overlay
        except Exception as e:
            print(f"⚠️ Grad-CAM malaria lỗi: {e}")
            return None

    # ── Build HTML report ─────────────────────────────────────────────────────

    def build_html_report(self, report: MalariaReport,
                          image_path: str = "") -> Tuple[str, str]:
        if report.prediction is None:
            return (
                "<span style='color:#dc2626;'>❌ Không thể phân loại. Kiểm tra model.</span>",
                "Lỗi: Không thể phân loại ảnh."
            )

        p = report.prediction
        rc = '#dc2626' if p.is_infected else '#16a34a'

        # Confidence bar
        bar_infected   = int(p.top2[0][1] * 140) if p.top2 else 0
        bar_uninfected = int(p.top2[1][1] * 140) if len(p.top2) > 1 else 0
        c_infected   = CLASS_INFO['Parasitized']['color']
        c_uninfected = CLASS_INFO['Uninfected']['color']

        bars_html = f"""
        <div style='margin:4px 0;'>
          <span style='display:inline-block;width:110px;font-weight:bold;
                color:{c_infected};font-size:12px;'>Parasitized</span>
          <span style='display:inline-block;width:{bar_infected}px;height:11px;
                background:{c_infected};border-radius:4px;vertical-align:middle;'></span>
          <span style='margin-left:6px;color:#101114;font-size:12px;'>
            {p.top2[0][1]*100:.1f}%</span>
        </div>
        <div style='margin:4px 0;'>
          <span style='display:inline-block;width:110px;font-weight:bold;
                color:{c_uninfected};font-size:12px;'>Uninfected</span>
          <span style='display:inline-block;width:{bar_uninfected}px;height:11px;
                background:{c_uninfected};border-radius:4px;vertical-align:middle;'></span>
          <span style='margin-left:6px;color:#101114;font-size:12px;'>
            {p.top2[1][1]*100:.1f}%</span>
        </div>""" if len(p.top2) > 1 else ""

        status_tag = (
            f"<span style='color:#dc2626;font-weight:bold;'>⚠ Phát hiện nhiễm sốt rét</span>"
            if p.is_infected else
            f"<span style='color:#16a34a;font-weight:bold;'>✓ Không phát hiện ký sinh trùng</span>"
        )

        html = f"""
        <div style='font-family:Segoe UI,Arial;font-size:13px;line-height:1.7;'>
          <div style='background:{p.color}18;border-left:4px solid {p.color};
               border-radius:0 10px 10px 0;padding:12px;margin-bottom:12px;'>
            <div style='font-size:20px;font-weight:900;color:{p.color};'>
              {p.icon} {p.description_vi}
            </div>
            <div style='font-size:13px;color:#101114;margin-top:4px;'>
              Độ tin cậy: <b>{p.confidence*100:.1f}%</b>
              &nbsp;·&nbsp; {status_tag}
            </div>
          </div>

          <div style='margin-bottom:8px;'>
            <b style='color:#101114;'>① Mức độ rủi ro:</b>
            <span style='color:{rc};font-weight:bold;font-size:15px;'>
              &nbsp;{p.risk_level}</span>
          </div>

          <div style='margin-bottom:8px;'>
            <b style='color:#101114;'>② Mô tả:</b><br>
            <span style='color:#5a5a72;'>{p.description_vi}</span><br>
            <span style='color:#5a5a72;'>{CLASS_INFO.get(p.predicted_class, {}).get("desc","")}</span>
          </div>

          <div style='margin-bottom:12px;'>
            <b style='color:#101114;'>③ Khuyến nghị:</b><br>
            <span style='color:#5a5a72;'>{p.recommendation}</span>
          </div>

          <div style='background:#f7f7fb;border-radius:10px;padding:10px 12px;'>
            <b style='color:#101114;font-size:12px;'>④ Phân phối xác suất:</b>
            <div style='margin-top:8px;'>{bars_html}</div>
          </div>

          <div style='margin-top:12px;padding:8px 10px;
               background:#fefce8;border-radius:8px;
               border-left:3px solid #d97706;'>
            <span style='color:#b45309;font-size:11px;'>
              ⚠ Kết quả AI hỗ trợ sàng lọc — cần xét nghiệm lâm sàng
              (giọt đặc, PCR) để chẩn đoán chính xác.
            </span>
          </div>
        </div>"""

        plain = (
            f"Kết quả: {p.description_vi} ({p.confidence*100:.1f}%)\n"
            f"Mức độ rủi ro: {p.risk_level}\n"
            f"{'Phát hiện nhiễm sốt rét' if p.is_infected else 'Không nhiễm ký sinh trùng'}\n"
            f"Khuyến nghị: {p.recommendation}\n"
            f"Lưu ý: Cần xét nghiệm lâm sàng để xác nhận."
        )
        return html, plain

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_input) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        raise ValueError(f"Kiểu không hỗ trợ: {type(image_input)}")