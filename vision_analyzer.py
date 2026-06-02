"""
vision_analyzer.py  (자세 + 표정 / 실시간 코칭 / 준비 카운트다운)
────────────────────────────────────────────────────────
YOLO Pose + MediaPipe FaceMesh 기반 실시간 면접 분석기

핵심 기능
  [자세 - YOLO Pose] 어깨 수평 / 고개 기울기 / 정면 응시 / 화면 위치·거리
  [표정·시선 - MediaPipe] 표정(밝음/무표정/굳음) / 시선 / 눈 깜빡임
  [공통]
    - 문제점 + 개선 코칭을 화면에 즉시 함께 표시
    - 시작 전 준비 카운트다운(기본 5초): 이 구간은 점수에 넣지 않음 ★이번 변경점
    - 종료 시 자세 + 표정 세션 요약 & 코칭

설치
  pip install ultralytics opencv-python mediapipe pillow numpy

실행
  python vision_analyzer.py
  (창에서 'q' 를 누르면 종료되고 요약이 출력됩니다)
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from collections import defaultdict
from math import atan2, degrees

import numpy as np

# ── 의존성 import ──────────────────────────────────────
try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python 이 필요합니다:  pip install opencv-python") from e

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics 가 필요합니다:  pip install ultralytics") from e

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe 가 필요합니다:  pip install mediapipe") from e

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── 설정 ────────────────────────────────────────────────
MODEL_NAME = "yolov8n-pose.pt"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0
MAX_ONSCREEN_ISSUES = 3
COUNTDOWN_SEC = 5            # ★ 준비 시간(초). 이 구간은 점수 집계 안 함. 바꾸려면 이 숫자만.

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH = 5, 6

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
R_EYE_TOP, R_EYE_BOT = 159, 145
R_EYE_L, R_EYE_R = 33, 133
L_EYE_TOP, L_EYE_BOT = 386, 374
L_EYE_L, L_EYE_R = 362, 263
R_IRIS, L_IRIS = 468, 473

ISSUE_LABELS = {
    "shoulder_tilt": "어깨가 한쪽으로 기울어졌어요",
    "head_tilt": "고개가 기울어졌어요",
    "not_facing": "정면을 바라보지 않고 있어요",
    "off_center": "화면 중앙에서 벗어났어요",
    "too_close": "카메라와 너무 가까워요",
    "too_far": "카메라와 너무 멀어요",
    "head_down": "고개가 떨어졌어요",
    "stiff_face": "표정이 굳어 있어요",
    "gaze_away": "시선이 정면을 벗어났어요",
    "blink_much": "눈을 너무 자주 깜빡여요",
}
SHORT_TIPS = {
    "shoulder_tilt": "양 어깨 높이를 맞추세요",
    "head_tilt": "고개를 똑바로 세우세요",
    "not_facing": "정면(카메라)을 바라보세요",
    "off_center": "화면 중앙으로 이동하세요",
    "too_close": "조금 뒤로 물러나세요",
    "too_far": "조금 앞으로 다가오세요",
    "head_down": "고개를 들고 시선을 위로",
    "stiff_face": "살짝 미소를 지어보세요",
    "gaze_away": "카메라에 시선을 두세요",
    "blink_much": "천천히 호흡하며 안정을",
}
ISSUE_TIPS = {
    "shoulder_tilt": "양쪽 어깨 높이를 맞추고 균형 있게 앉으세요.",
    "head_tilt": "고개를 똑바로 세워 화면과 수평을 맞추세요.",
    "not_facing": "면접관(카메라)을 정면으로 바라보세요.",
    "off_center": "상반신이 화면 중앙에 오도록 위치를 잡으세요.",
    "too_close": "카메라에서 조금 떨어져 상반신이 다 보이게 하세요.",
    "too_far": "카메라에 조금 더 가까이 앉으세요.",
    "head_down": "고개를 들고 시선을 정면으로 향하세요.",
    "stiff_face": "면접에서는 자연스러운 미소가 호감을 줍니다. 너무 굳지 않게 연습하세요.",
    "gaze_away": "답변 중에도 카메라(면접관)에 시선을 두는 연습을 하세요.",
    "blink_much": "긴장하면 눈 깜빡임이 늘어납니다. 천천히 호흡하며 안정을 찾으세요.",
}
ISSUE_PRIORITY = [
    "head_down", "not_facing", "gaze_away", "stiff_face",
    "shoulder_tilt", "head_tilt", "off_center", "too_close", "too_far", "blink_much",
]


class VisionAnalyzer:
    """YOLO Pose(자세) + MediaPipe FaceMesh(표정·시선) 통합 실시간 분석기"""

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            print(f"[안내] 자세 모델 로딩 중... ({model_name})  첫 실행이면 다운로드가 진행됩니다.")
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(
                f"YOLO 모델 로드 실패: {e}\n"
                "  · 인터넷 연결을 확인하세요(첫 실행 시 모델 다운로드 필요).\n"
                "  · 계속 실패하면 코드 상단 MODEL_NAME 을 'yolo11n-pose.pt' 로 바꿔보세요."
            ) from e

        print("[안내] 표정 분석기(MediaPipe FaceMesh) 초기화 중...")
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        self.font_huge = self._load_font(140)   # ★ 카운트다운 큰 숫자용
        self.font_big = self._load_font(32)
        self.font = self._load_font(21)
        self.font_tip = self._load_font(17)

        self.frame_count = 0
        self.score_sum = 0
        self.issue_counts = defaultdict(int)
        self.expr_counts = defaultdict(int)

    def _load_font(self, size: int):
        if not PIL_OK:
            return None
        for path in [FONT_PATH, "C:/Windows/Fonts/malgunbd.ttf",
                     "C:/Windows/Fonts/gulim.ttc", "C:/Windows/Fonts/batang.ttc"]:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return None

    # ── 자세 분석 (YOLO) ─────────────────────────────────
    def _extract_keypoints(self, result):
        kp = result.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return None, None
        pts = kp.xy[0].cpu().numpy()
        cfs = kp.conf[0].cpu().numpy() if kp.conf is not None else None
        return pts, cfs

    def _pt(self, pts, cfs, idx):
        if cfs is not None and cfs[idx] < CONF_THRESHOLD:
            return None
        x, y = pts[idx]
        if x == 0 and y == 0:
            return None
        return np.array([float(x), float(y)])

    @staticmethod
    def _tilt_deg(p1, p2) -> float:
        d = p2 - p1
        return degrees(atan2(abs(d[1]), abs(d[0]) + 1e-6))

    def analyze_posture(self, pts, cfs, frame_w: int) -> dict:
        if pts is None:
            return {"detected": False, "issues": []}
        l_sh = self._pt(pts, cfs, L_SH)
        r_sh = self._pt(pts, cfs, R_SH)
        nose = self._pt(pts, cfs, NOSE)
        l_eye = self._pt(pts, cfs, L_EYE)
        r_eye = self._pt(pts, cfs, R_EYE)
        if l_sh is None or r_sh is None:
            return {"detected": False, "issues": []}

        issues = []
        sh_mid = (l_sh + r_sh) / 2.0
        sh_width = float(np.linalg.norm(l_sh - r_sh)) + 1e-6

        if self._tilt_deg(l_sh, r_sh) > 8.0:
            issues.append("shoulder_tilt")
        if l_eye is not None and r_eye is not None:
            if self._tilt_deg(l_eye, r_eye) > 8.0:
                issues.append("head_tilt")
        if nose is not None:
            nose_offset = (nose[0] - sh_mid[0]) / sh_width
            facing_bad = abs(nose_offset) > 0.22
            if l_eye is not None and r_eye is not None:
                dl = np.linalg.norm(nose - l_eye)
                dr = np.linalg.norm(nose - r_eye)
                if min(dl, dr) / (max(dl, dr) + 1e-6) < 0.55:
                    facing_bad = True
            if facing_bad:
                issues.append("not_facing")
            if (sh_mid[1] - nose[1]) / sh_width < 0.35:
                issues.append("head_down")

        center_x = sh_mid[0] / frame_w
        if center_x < 0.35 or center_x > 0.65:
            issues.append("off_center")
        width_ratio = sh_width / frame_w
        if width_ratio > 0.55:
            issues.append("too_close")
        elif width_ratio < 0.18:
            issues.append("too_far")

        return {"detected": True, "issues": issues}

    # ── 표정·시선 분석 (MediaPipe) ───────────────────────
    @staticmethod
    def _lm_xy(landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def analyze_face(self, image_rgb, w: int, h: int) -> dict:
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return {"detected": False, "expression": "-", "issues": []}

        lms = results.multi_face_landmarks[0].landmark
        issues = []
        eye_l = self._lm_xy(lms, R_EYE_L, w, h)
        eye_r = self._lm_xy(lms, L_EYE_R, w, h)
        face_w = float(np.linalg.norm(eye_l - eye_r)) + 1e-6

        m_left = self._lm_xy(lms, MOUTH_LEFT, w, h)
        m_right = self._lm_xy(lms, MOUTH_RIGHT, w, h)
        m_top = self._lm_xy(lms, MOUTH_TOP, w, h)
        m_bot = self._lm_xy(lms, MOUTH_BOTTOM, w, h)
        smile_ratio = (((m_top[1] + m_bot[1]) / 2.0) - ((m_left[1] + m_right[1]) / 2.0)) / face_w

        if smile_ratio > 0.02:
            expression = "밝음"
        elif smile_ratio > -0.01:
            expression = "무표정"
        else:
            expression = "굳음"
            issues.append("stiff_face")

        def eye_open_ratio(top, bot, left, right):
            v = abs(self._lm_xy(lms, top, w, h)[1] - self._lm_xy(lms, bot, w, h)[1])
            hgt = abs(self._lm_xy(lms, left, w, h)[0] - self._lm_xy(lms, right, w, h)[0]) + 1e-6
            return v / hgt
        ear = (eye_open_ratio(R_EYE_TOP, R_EYE_BOT, R_EYE_L, R_EYE_R) +
               eye_open_ratio(L_EYE_TOP, L_EYE_BOT, L_EYE_L, L_EYE_R)) / 2.0
        is_closed = ear < 0.18
        if is_closed and not self._eye_was_closed:
            self._blink_count += 1
        self._eye_was_closed = is_closed

        try:
            r_iris = self._lm_xy(lms, R_IRIS, w, h)
            r_in = self._lm_xy(lms, R_EYE_L, w, h)
            r_out = self._lm_xy(lms, R_EYE_R, w, h)
            denom = (r_out[0] - r_in[0])
            if abs(denom) > 1e-6:
                gaze_pos = (r_iris[0] - r_in[0]) / denom
                if gaze_pos < 0.30 or gaze_pos > 0.70:
                    issues.append("gaze_away")
        except Exception:
            pass

        return {"detected": True, "expression": expression, "issues": issues}

    _eye_was_closed = False
    _blink_count = 0

    # ── 점수 + 화면 ──────────────────────────────────────
    @staticmethod
    def _score_from_issues(n_issues: int) -> int:
        return max(0, 100 - 14 * n_issues)

    @staticmethod
    def _sort_by_priority(issues):
        order = {k: i for i, k in enumerate(ISSUE_PRIORITY)}
        return sorted(issues, key=lambda k: order.get(k, 999))

    def _draw_countdown(self, frame, seconds_left: int):
        """★ 준비 카운트다운 화면: 큰 숫자 + 안내. 점수는 집계하지 않는 구간."""
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        if not PIL_OK or self.font_huge is None:
            cv2.putText(frame, str(seconds_left), (W // 2 - 40, H // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 6)
            cv2.putText(frame, "Get ready...", (W // 2 - 120, H // 2 + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        num = str(seconds_left)
        # 큰 숫자 중앙 정렬
        bbox = draw.textbbox((0, 0), num, font=self.font_huge)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((W - tw) / 2, (H - th) / 2 - 40), num,
                  font=self.font_huge, fill=(120, 220, 255))
        # 안내 문구
        msg = "잠시 후 분석을 시작합니다"
        b2 = draw.textbbox((0, 0), msg, font=self.font_big)
        draw.text(((W - (b2[2] - b2[0])) / 2, H / 2 + 80), msg,
                  font=self.font_big, fill=(230, 230, 230))
        sub = "자세를 바르게 하고 카메라를 정면으로 바라보세요"
        b3 = draw.textbbox((0, 0), sub, font=self.font)
        draw.text(((W - (b3[2] - b3[0])) / 2, H / 2 + 124), sub,
                  font=self.font, fill=(180, 180, 180))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_feedback(self, frame, posture, face, score):
        H, W = frame.shape[:2]
        detected = posture["detected"] or face["detected"]
        issues = self._sort_by_priority(posture["issues"] + face["issues"])
        expression = face["expression"] if face["detected"] else "-"

        shown = issues[:MAX_ONSCREEN_ISSUES]
        hidden_n = len(issues) - len(shown)

        header_h = 86
        block_h = 50
        extra = 24 if hidden_n > 0 else 0
        body_h = (block_h * len(shown)) if shown else 30
        panel_h = header_h + body_h + extra

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (min(540, W), panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        if not detected:
            score_color = (180, 180, 180)
        elif score >= 80:
            score_color = (90, 230, 120)
        elif score >= 60:
            score_color = (255, 210, 70)
        else:
            score_color = (255, 90, 90)

        if self.font_big is None or not PIL_OK:
            txt = "No person" if not detected else f"Score:{score} Expr:{expression}"
            bgr = (score_color[2], score_color[1], score_color[0])
            cv2.putText(frame, txt, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        if not detected:
            draw.text((14, 10), "분석 대기 --", font=self.font_big, fill=score_color)
            draw.text((14, 50), "사람이 감지되지 않았습니다", font=self.font, fill=(255, 190, 0))
        else:
            draw.text((14, 8), f"종합 점수 {score}", font=self.font_big, fill=score_color)
            expr_color = {"밝음": (90, 230, 120), "무표정": (255, 210, 70),
                          "굳음": (255, 120, 120)}.get(expression, (200, 200, 200))
            draw.text((14, 50), f"표정: {expression}", font=self.font, fill=expr_color)

            if not shown:
                draw.text((14, 80), "OK 자세·표정 모두 안정적입니다",
                          font=self.font, fill=(90, 230, 120))
            else:
                y = 82
                for key in shown:
                    draw.text((14, y), "● " + ISSUE_LABELS.get(key, key),
                              font=self.font, fill=(255, 110, 110))
                    draw.text((34, y + 24), "→ " + SHORT_TIPS.get(key, ""),
                              font=self.font_tip, fill=(150, 220, 255))
                    y += block_h
                if hidden_n > 0:
                    draw.text((14, y), f"… 외 {hidden_n}개 더 (요약에서 전체 확인)",
                              font=self.font_tip, fill=(190, 190, 190))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ── 메인 루프 ────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[에러] 웹캠을 열 수 없습니다. 카메라 연결/권한을 확인하세요.")
            return

        print(f"\n[안내] {COUNTDOWN_SEC}초 후 분석을 시작합니다. (이 준비 구간은 점수에 반영되지 않습니다)")
        print("[안내] 창에서 'q' 를 누르면 종료됩니다.\n")
        prev_t = time.time()
        start_t = time.time()       # ★ 시작 시각 — 카운트다운 기준
        analyzing_started = False    # ★ 본 분석 시작 알림 1회용 플래그

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[경고] 프레임을 읽지 못했습니다.")
                    break

                frame_h, frame_w = frame.shape[:2]
                elapsed = time.time() - start_t          # ★ 경과 시간
                in_countdown = elapsed < COUNTDOWN_SEC    # ★ 준비 구간 여부

                # 자세·표정은 준비 구간에도 화면에 보여줌(미리보기). 단 점수는 집계 안 함.
                try:
                    results = self.model(frame, verbose=False)
                    result = results[0]
                    annotated = result.plot()
                    pts, cfs = self._extract_keypoints(result)
                except Exception as e:
                    print(f"[경고] 자세 추론 오류: {e}")
                    annotated, pts, cfs = frame.copy(), None, None

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = self.analyze_face(image_rgb, frame_w, frame_h)
                except Exception as e:
                    print(f"[경고] 표정 분석 오류: {e}")
                    face = {"detected": False, "expression": "-", "issues": []}

                posture = self.analyze_posture(pts, cfs, frame_w)
                all_issues = posture["issues"] + face["issues"]
                detected = posture["detected"] or face["detected"]
                score = self._score_from_issues(len(all_issues)) if detected else 0

                if in_countdown:
                    # ★ 준비 구간: 점수 집계 없이 카운트다운만 표시
                    seconds_left = int(COUNTDOWN_SEC - elapsed) + 1
                    annotated = self._draw_countdown(annotated, seconds_left)
                else:
                    # ★ 본 분석 구간: 통계 누적 + 피드백 표시
                    if not analyzing_started:
                        print("[안내] 분석을 시작합니다!\n")
                        analyzing_started = True
                    if detected:
                        self.frame_count += 1
                        self.score_sum += score
                        for key in all_issues:
                            self.issue_counts[key] += 1
                        if face["detected"]:
                            self.expr_counts[face["expression"]] += 1
                    annotated = self.draw_feedback(annotated, posture, face, score)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                cv2.putText(annotated, f"FPS {fps:4.1f}",
                            (annotated.shape[1] - 120, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("AI Interview Coach - Posture & Face (q to quit)", annotated)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\n[안내] 사용자 중단(Ctrl+C).")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.print_summary()

    # ── 세션 요약 ────────────────────────────────────────
    def print_summary(self):
        print("\n" + "=" * 52)
        print("  자세 + 표정 분석 세션 요약")
        print("=" * 52)
        if self.frame_count == 0:
            print("  분석된 프레임이 없습니다. (얼굴/어깨가 화면에 잡혔는지 확인하세요)")
            print("=" * 52)
            return

        avg = self.score_sum / self.frame_count
        grade = self._to_grade(avg)
        print(f"  종합 평균 점수: {avg:.1f}점   (등급: {grade})")
        print(f"  분석 프레임 수: {self.frame_count}  (준비 {COUNTDOWN_SEC}초 제외)")
        print(f"  총 눈 깜빡임 수: {self._blink_count}회")

        if self.expr_counts:
            print("\n  [표정 분포]")
            total_expr = sum(self.expr_counts.values())
            for label, cnt in sorted(self.expr_counts.items(), key=lambda x: -x[1]):
                print(f"   - {label}: 전체의 {cnt / total_expr * 100:.0f}%")

        if self.issue_counts:
            print("\n  [자주 나타난 문제]")
            for key, cnt in sorted(self.issue_counts.items(), key=lambda x: -x[1]):
                print(f"   - {ISSUE_LABELS.get(key, key)}  ->  전체의 {cnt / self.frame_count * 100:.0f}% 구간")
            print("\n  [개선 코칭]")
            for key, _ in sorted(self.issue_counts.items(), key=lambda x: -x[1]):
                print(f"   * {ISSUE_TIPS.get(key, '-')}")
        else:
            print("\n  OK 자세와 표정 모두 전반적으로 안정적이었습니다. 훌륭해요!")
        print("=" * 52)

    @staticmethod
    def _to_grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"


if __name__ == "__main__":
    try:
        analyzer = VisionAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"[에러] 실행 실패: {e}")
