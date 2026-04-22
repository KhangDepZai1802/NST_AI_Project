# src/core/ai_model.py
import os
import cv2
from ultralytics import YOLO
from src.core.chromosome_classifier import ChromosomeClassifier, AnalysisReport
from src.core.resource_path import rp

class ChromosomeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.classifier = None
        self.load_model()

    def _try_load_yolo(self, path: str):
        if not path or not str(path).strip():
            return None
        # Thử tìm file bằng đường dẫn tuyệt đối của PyInstaller
        resolved = path if os.path.isfile(path) else rp(path)
        
        if not os.path.isfile(resolved):
            print(f"⚠️ Không tìm thấy model tại: {resolved}")
            return None
            
        # Dùng 'resolved' thay cho 'path' để load model
        try:
            return YOLO(resolved) # Hoặc câu lệnh load model tương ứng của bạn
        except Exception as e:
            print(f"Lỗi load model: {e}")
            return None

    def load_model(self):
        # YOLO phân đoạn
        self.model = self._try_load_yolo(
            self.config.get("model_path", "models/best.pt")
        )

        # ResNet classifier (chỉ dùng để detect chX)
        resnet_path = self.config.get(
            "model_classifier_path", "models/best_CirNET_v2.pth"
        )
        if self.classifier is None or \
                getattr(self.classifier, "model_path", "") != resnet_path:
            self.classifier = ChromosomeClassifier(model_path=resnet_path)
        else:
            self.classifier.reload(resnet_path)

        if self.model:
            print("✅ YOLO sẵn sàng.")
        else:
            print("❌ Chưa load được YOLO.")

    def analyze(self, image_path: str):
        if self.model is None:
            raise Exception(
                "YOLO model chưa load! Kiểm tra đường dẫn trong Cài đặt."
            )

        normal    = int(self.config.get("normal_count", 46))
        tolerance = int(self.config.get("tolerance", 1))

        orig_bgr = cv2.imread(image_path)
        results  = self.model.predict(source=image_path, conf=0.25, save=False)
        result   = results[0]

        annotated_img = result.plot()
        total_count   = len(result.boxes)

        report: AnalysisReport = self.classifier.analyze(
            result, normal=normal, tolerance=tolerance, orig_bgr=orig_bgr
        )

        report_html, report_plain = self._build_reports(
            report, total_count, normal, tolerance
        )
        return annotated_img, total_count, report_html, report_plain, report

    # ── Build reports ─────────────────────────────────────────────────────────

    def _build_reports(self, r: AnalysisReport, total, normal, tolerance):
        diff        = abs(total - normal)
        count_color = "#27AE60" if r.is_normal_count else "#C0392B"
        count_label = "Bình thường" if r.is_normal_count else (
            f"Bất thường — {'thừa' if total > normal else 'thiếu'} {diff} NST"
        )

        risk_colors = {
            "Bình thường":   "#27AE60",
            "Cần theo dõi":  "#E67E22",
            "Nguy cơ cao":   "#C0392B",
        }
        risk_color = risk_colors.get(r.risk_level, "#7F8C8D")

        # Nhóm Denver — hiển thị so sánh với kỳ vọng
        _EXPECTED = {"A": 6, "B": 4, "C": 16, "D": 6, "E": 6, "F": 4, "G": 4}
        group_parts = []
        for g in "ABCDEFG":
            n   = r.group_sizes.get(g, 0)
            exp = _EXPECTED.get(g, 0)
            if n != exp:
                color = "#C0392B" if abs(n - exp) > 1 else "#E67E22"
                group_parts.append(
                    f"<b>{g}</b>:<span style='color:{color};'>{n}</span>"
                    f"<span style='color:#95A5A6;font-size:11px;'>(kỳ vọng {exp})</span>"
                )
            else:
                group_parts.append(f"<b>{g}</b>:{n}")
        group_str = "  ".join(group_parts)

        # Hội chứng
        if r.syndrome_flags:
            syndrome_html  = "<ul style='margin:4px 0 0 16px;padding:0;'>" + \
                "".join(f"<li style='color:#C0392B;'>{s}</li>"
                        for s in r.syndrome_flags) + "</ul>"
            syndrome_plain = "\n".join(f"  • {s}" for s in r.syndrome_flags)
        else:
            syndrome_html  = "<span style='color:#27AE60;'>Không phát hiện hội chứng đặc trưng</span>"
            syndrome_plain = "Không phát hiện hội chứng đặc trưng"

        # Tag phương pháp
        method_tag = (
            "<span style='color:#2ECC71;font-size:11px;'>"
            "Karyotype: YOLO + Denver (kích thước mask)"
            + (" + ResNet chX" if r.classifier_active and
               any(f.resnet_label for f in r.features) else "")
            + "</span>"
        )

        html_lines = [
            f"<b>① Số lượng NST:</b> "
            f"<span style='font-size:20px;font-weight:900;color:{count_color};'>"
            f"{total}</span>",

            f"<b>② Đánh giá:</b> "
            f"<span style='color:{count_color};'>"
            f"{count_label} (chuẩn {normal} ± {tolerance})</span>",

            f"<b>③ Giới tính ước tính:</b> {r.sex_estimation} "
            f"<span style='color:#7F8C8D;font-size:12px;'>"
            f"(Độ tin cậy: {r.sex_confidence})</span>",

            f"<b>④ Mức độ rủi ro:</b> "
            f"<span style='color:{risk_color};font-weight:bold;'>"
            f"{r.risk_level}</span>",

            f"<b>⑤ Phân nhóm Denver:</b><br>"
            f"<span style='font-size:12px;'>{group_str}</span>",

            f"<b>⑥ Hội chứng nghi ngờ:</b>{syndrome_html}",

            method_tag,

            "<span style='color:#95A5A6;font-size:11px;'>"
            "⚠ Kết quả hỗ trợ — cần karyotype chuyên sâu để xác nhận.</span>",
        ]

        plain_lines = [
            f"1. Số NST: {total}",
            f"2. Đánh giá: {count_label} (chuẩn {normal} ± {tolerance})",
            f"3. Giới tính: {r.sex_estimation} (Độ tin cậy: {r.sex_confidence})",
            f"4. Rủi ro: {r.risk_level}",
            "5. Phân nhóm Denver: " + ", ".join(
                f"{g}:{r.group_sizes.get(g,0)}" for g in "ABCDEFG"
            ),
            f"6. Hội chứng:\n{syndrome_plain}",
            "Lưu ý: Kết quả hỗ trợ chẩn đoán, không thay thế xét nghiệm.",
        ]

        return "<br>".join(html_lines), "\n".join(plain_lines)