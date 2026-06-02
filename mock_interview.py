"""
mock_interview.py  (완전한 모의면접 / 5문항 연속 + 종합 리포트)
────────────────────────────────────────────────────────
여러 질문을 연속 출제하는 본격 모의면접.

흐름
  질문 N개 무작위 선택(중복 없음)
   → [문항마다] 준비 카운트다운 → 답변(자세·표정 측정 + 음성 녹음 동시)
                → 'q'로 답변 종료 → 음성→글자 → 내용 평가
   → 문항 사이 전환 화면
   → 전체 종합 리포트(문항별 점수 + 평균 + 강·약 문항)
   → 세션 평균을 성장추적에 1회차로 저장 + 곡선 갱신

조작
  · 답변 끝: 카메라 창에서 'q'
  · 세션 중단: 카메라 창에서 ESC

합쳐진 모듈
  - YOLO 자세 + MediaPipe 표정·시선  (이 파일 내장)
  - content_evaluator.py : Ollama 내용 평가  (같은 폴더, 선택)
  - growth_tracker.py    : 성장 추적         (같은 폴더, 선택)
  - Whisper              : 음성→글자(STT)

사전 준비
  pip install ultralytics opencv-python mediapipe pillow numpy openai-whisper sounddevice scipy matplotlib
  winget install ffmpeg
  (내용 평가) ollama 실행 + ollama pull qwen2.5:7b + pip install ollama

실행
  python mock_interview.py
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import queue
import random
import tempfile
from collections import defaultdict, deque

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
    import sounddevice as sd
except ImportError as e:
    raise ImportError("sounddevice 가 필요합니다:  pip install sounddevice") from e
try:
    from scipy.io.wavfile import write as wav_write
except ImportError as e:
    raise ImportError("scipy 가 필요합니다:  pip install scipy") from e
try:
    import whisper
except ImportError as e:
    raise ImportError("openai-whisper 가 필요합니다:  pip install openai-whisper") from e
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

# content_evaluator (선택)
try:
    from content_evaluator import ContentEvaluator
    EVALUATOR_AVAILABLE = True
except Exception as _e:
    EVALUATOR_AVAILABLE = False
    _EVALUATOR_IMPORT_ERR = _e

# growth_tracker (선택)
try:
    import growth_tracker
    GROWTH_AVAILABLE = True
except Exception as _ge:
    GROWTH_AVAILABLE = False
    _GROWTH_IMPORT_ERR = _ge

# report_generator (선택): 있으면 면접 종료 시 HTML 리포트 생성 + 브라우저 열기
try:
    import report_generator
    REPORT_AVAILABLE = True
except Exception as _re:
    REPORT_AVAILABLE = False
    _REPORT_IMPORT_ERR = _re

# question_bank (선택): 있으면 직무별 맞춤 질문 사용. 없으면 아래 QUESTIONS 사용.
try:
    import question_bank
    QBANK_AVAILABLE = True
except Exception as _qe:
    QBANK_AVAILABLE = False
    _QBANK_IMPORT_ERR = _qe


# ── 설정 ────────────────────────────────────────────────
MODEL_NAME = "yolov8n-pose.pt"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0
MAX_ONSCREEN_ISSUES = 3
COUNTDOWN_SEC = 5
NUM_QUESTIONS = 5            # ★ 한 세션 문항 수
TRANSITION_SEC = 3          # ★ 문항 사이 전환 화면 시간(초)

SAMPLE_RATE = 16000
WHISPER_MODEL = "medium"    # 정확도 최고. 처음 1.5GB 다운로드, 변환은 다소 느림.
                            #  더 빠르게: "small"(균형) 또는 "base"(빠름)
EXPR_SMOOTH_FRAMES = 15
MOUTH_OPEN_TALKING = 0.06
JOB_ROLE = "일반 직무"

QUESTIONS = [
    "간단하게 자기소개를 해주세요.",
    "본인의 가장 큰 강점은 무엇이며, 그것을 발휘한 경험을 말해주세요.",
    "지원한 직무에 본인이 적합하다고 생각하는 이유는 무엇인가요?",
    "최근에 어려운 문제를 해결했던 경험을 구체적으로 말해주세요.",
    "5년 후 본인의 모습을 어떻게 그리고 있나요?",
    "팀에서 갈등이 있었을 때 어떻게 해결했는지 말해주세요.",
    "우리 회사(직무)에 지원한 동기는 무엇인가요?",
    "본인의 단점은 무엇이며, 그것을 극복하기 위해 어떤 노력을 하나요?",
    "지금까지 가장 큰 성취 경험은 무엇인가요?",
    "스트레스를 받을 때 어떻게 관리하나요?",
]

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH = 5, 6
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
R_EYE_TOP, R_EYE_BOT = 159, 145
R_EYE_L, R_EYE_R = 33, 133
L_EYE_TOP, L_EYE_BOT = 386, 374
L_EYE_L, L_EYE_R = 362, 263
R_IRIS = 468

ISSUE_LABELS = {
    "shoulder_tilt": "어깨가 한쪽으로 기울어졌어요",
    "head_tilt": "고개가 기울어졌어요",
    "not_facing": "정면을 바라보지 않고 있어요",
    "off_center": "화면 중앙에서 벗어났어요",
    "too_close": "카메라와 너무 가까워요",
    "too_far": "카메라와 너무 멀어요",
    "head_down": "고개가 떨어졌어요",
    "stiff_face": "표정이 굳어 있어요",
    "fake_smile": "입만 웃는 어색한 미소예요",
    "gaze_away": "시선이 정면을 벗어났어요",
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
    "fake_smile": "눈도 함께 웃어보세요",
    "gaze_away": "카메라에 시선을 두세요",
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
    "fake_smile": "입꼬리만 올리면 어색해 보입니다. 눈가까지 자연스럽게 풀어 진짜 미소를 지어보세요.",
    "gaze_away": "답변 중에도 카메라(면접관)에 시선을 두는 연습을 하세요.",
}
ISSUE_PRIORITY = [
    "head_down", "not_facing", "gaze_away", "stiff_face", "fake_smile",
    "shoulder_tilt", "head_tilt", "off_center", "too_close", "too_far",
]


class MockInterview:
    """여러 문항을 연속 진행하는 모의면접"""

    def __init__(self, job_role: str = "일반 직무"):
        self.job_role = job_role or "일반 직무"   # ★ 지원 직무(내용 평가 기준)
        print(f"[안내] 자세 모델 로딩 중... ({MODEL_NAME})  첫 실행이면 다운로드가 진행됩니다.")
        try:
            self.model = YOLO(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"YOLO 모델 로드 실패: {e}") from e

        print("[안내] 표정 분석기(MediaPipe) 초기화 중...")
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        # Whisper 는 첫 문항 변환 때 한 번만 로드해서 재사용
        self._whisper_model = None

        # 폰트
        self.font_huge = self._load_font(140)
        self.font_big = self._load_font(30)
        self.font = self._load_font(20)
        self.font_tip = self._load_font(16)
        self.font_q = self._load_font(23)

        # 오디오
        self._audio_q = queue.Queue()
        self._audio_chunks = []

        # 깜빡임/표정 상태(문항마다 reset)
        self._eye_was_closed = False
        self._blink_count = 0
        self._expr_history = deque(maxlen=EXPR_SMOOTH_FRAMES)
        self._ear_samples = deque(maxlen=150)   # 평소 눈 크기 기준선용(최근 프레임)

    def _load_font(self, size):
        if not PIL_OK:
            return None
        for path in [FONT_PATH, "C:/Windows/Fonts/malgunbd.ttf",
                     "C:/Windows/Fonts/gulim.ttc", "C:/Windows/Fonts/batang.ttc"]:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return None

    def _reset_question_state(self):
        """문항 시작 시 통계 초기화."""
        self.frame_count = 0
        self.score_sum = 0
        self.issue_counts = defaultdict(int)
        self.expr_counts = defaultdict(int)
        self._eye_was_closed = False
        self._blink_count = 0
        self._expr_history.clear()
        self._ear_samples.clear()   # 평소 눈 크기(EAR) 기준선 표본
        self._audio_chunks = []
        # 큐 비우기
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except Exception:
                break

    # ════════════════════════════════════════════════════
    #  자세 분석
    # ════════════════════════════════════════════════════
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
    def _tilt_deg(p1, p2):
        from math import atan2, degrees
        d = p2 - p1
        return degrees(atan2(abs(d[1]), abs(d[0]) + 1e-6))

    def analyze_posture(self, pts, cfs, frame_w):
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
            if abs((nose[0] - sh_mid[0]) / sh_width) > 0.22:
                issues.append("not_facing")
            elif l_eye is not None and r_eye is not None:
                dl = np.linalg.norm(nose - l_eye)
                dr = np.linalg.norm(nose - r_eye)
                if min(dl, dr) / (max(dl, dr) + 1e-6) < 0.55:
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
        return {"detected": True, "issues": list(dict.fromkeys(issues))}

    # ════════════════════════════════════════════════════
    #  표정·시선 분석
    # ════════════════════════════════════════════════════
    @staticmethod
    def _lm_xy(landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def analyze_face(self, image_rgb, w, h):
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            self._expr_history.append("무표정")
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
        mouth_open = (m_bot[1] - m_top[1]) / face_w
        talking = mouth_open > MOUTH_OPEN_TALKING

        # ── 눈 뜬 정도(EAR) 먼저 계산: 깜빡임 + 표정 판정에 함께 사용 ──
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

        # ── 평소(중립) 눈 크기 기준선 추정: 깜빡임이 아닌 프레임의 EAR을 누적 평균 ──
        # 진짜 미소는 눈가 근육이 수축해 눈이 살짝 '가늘어진다'(EAR 감소).
        # 입만 웃고 눈은 그대로면 '어색한 미소'로 본다.
        if not is_closed and 0.18 <= ear <= 0.45:
            self._ear_samples.append(ear)
        if self._ear_samples:
            ear_baseline = sum(self._ear_samples) / len(self._ear_samples)
        else:
            ear_baseline = 0.30  # 표본 모이기 전 기본값

        # 눈가가 자연스럽게 좁아졌는지(진짜 미소 신호). 기준선 대비 5% 이상 감소.
        eyes_engaged = (not is_closed) and (ear < ear_baseline * 0.95)
        # 눈을 과하게 부릅뜸(놀람/긴장) 또는 과하게 찡그림(어색) 감지
        eyes_wide = ear > ear_baseline * 1.25
        eyes_squint = (not is_closed) and (ear < ear_baseline * 0.65)

        # ── 표정 판정: 입 + 눈을 함께 본다 ──
        if talking:
            instant_expr = "무표정"          # 말하는 중엔 표정 판정 보류
        elif smile_ratio > 0.015:
            # 입꼬리가 올라감 → 눈이 따라오는지 확인
            if eyes_squint:
                instant_expr = "어색"        # 입은 웃는데 눈을 찡그림 → 부자연
            elif eyes_engaged:
                instant_expr = "밝음"        # 입+눈 모두 웃음 → 진짜 미소
            else:
                instant_expr = "어색"        # 입만 웃고 눈은 그대로 → 가짜 미소
        elif smile_ratio > -0.025:
            instant_expr = "굳음" if eyes_wide else "무표정"
        else:
            instant_expr = "굳음"

        self._expr_history.append(instant_expr)
        expression = max(set(self._expr_history), key=self._expr_history.count)
        if expression == "굳음" and len(self._expr_history) >= EXPR_SMOOTH_FRAMES // 2:
            issues.append("stiff_face")
        elif expression == "어색" and len(self._expr_history) >= EXPR_SMOOTH_FRAMES // 2:
            issues.append("fake_smile")

        try:
            r_iris = self._lm_xy(lms, R_IRIS, w, h)
            r_in = self._lm_xy(lms, R_EYE_L, w, h)
            r_out = self._lm_xy(lms, R_EYE_R, w, h)
            denom = (r_out[0] - r_in[0])
            if abs(denom) > 1e-6:
                gaze_pos = (r_iris[0] - r_in[0]) / denom
                if gaze_pos < 0.28 or gaze_pos > 0.72:
                    issues.append("gaze_away")
        except Exception:
            pass
        return {"detected": True, "expression": expression, "issues": issues}

    # ════════════════════════════════════════════════════
    #  점수 + 화면
    # ════════════════════════════════════════════════════
    @staticmethod
    def _score_from_issues(n):
        return max(0, 100 - 14 * n)

    @staticmethod
    def _sort_by_priority(issues):
        order = {k: i for i, k in enumerate(ISSUE_PRIORITY)}
        return sorted(issues, key=lambda k: order.get(k, 999))

    @staticmethod
    def _draw_wrapped(draw, text, x, y, max_w, font, fill):
        words = text.split(" ")
        line = ""
        cy = y
        for w in words:
            test = (line + " " + w).strip()
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] > max_w and line:
                draw.text((x, cy), line, font=font, fill=fill)
                cy += (bb[3] - bb[1]) + 8
                line = w
            else:
                line = test
        if line:
            draw.text((x, cy), line, font=font, fill=fill)
        return cy

    def _draw_countdown(self, frame, seconds_left, question, q_no, q_total):
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        if not PIL_OK or self.font_huge is None:
            cv2.putText(frame, str(seconds_left), (W // 2 - 40, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 6)
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.text((40, 30), f"질문 {q_no} / {q_total}", font=self.font, fill=(170, 170, 170))
        self._draw_wrapped(d, "Q. " + question, 40, 64, W - 80, self.font_q, (255, 230, 130))
        num = str(seconds_left)
        bb = d.textbbox((0, 0), num, font=self.font_huge)
        d.text(((W - (bb[2] - bb[0])) / 2, (H - (bb[3] - bb[1])) / 2), num,
               font=self.font_huge, fill=(120, 220, 255))
        msg = "잠시 후 답변을 시작하세요"
        b2 = d.textbbox((0, 0), msg, font=self.font_big)
        d.text(((W - (b2[2] - b2[0])) / 2, H / 2 + 110), msg, font=self.font_big, fill=(230, 230, 230))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _draw_transition(self, frame, seconds_left, next_no, q_total):
        """문항 사이 전환 화면."""
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        if not PIL_OK or self.font_big is None:
            cv2.putText(frame, f"Next Q in {seconds_left}", (W // 2 - 150, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        msg = f"다음 질문 ({next_no}/{q_total}) 준비"
        b = d.textbbox((0, 0), msg, font=self.font_big)
        d.text(((W - (b[2] - b[0])) / 2, H / 2 - 60), msg, font=self.font_big, fill=(230, 230, 230))
        num = str(seconds_left)
        bb = d.textbbox((0, 0), num, font=self.font_huge)
        d.text(((W - (bb[2] - bb[0])) / 2, H / 2 - 10), num, font=self.font_huge, fill=(120, 220, 255))
        sub = "잠시 숨을 고르세요"
        b3 = d.textbbox((0, 0), sub, font=self.font)
        d.text(((W - (b3[2] - b3[0])) / 2, H / 2 + 140), sub, font=self.font, fill=(170, 170, 170))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_feedback(self, frame, posture, face, score, question, q_no, q_total):
        H, W = frame.shape[:2]
        detected = posture["detected"] or face["detected"]
        issues = self._sort_by_priority(posture["issues"] + face["issues"])
        expression = face["expression"] if face["detected"] else "-"
        shown = issues[:MAX_ONSCREEN_ISSUES]
        hidden_n = len(issues) - len(shown)

        qbar_h = 56
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, qbar_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)

        header_h = 60
        block_h = 48
        extra = 22 if hidden_n > 0 else 0
        body_h = (block_h * len(shown)) if shown else 28
        panel_top = qbar_h + 8
        panel_h = header_h + body_h + extra
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, panel_top), (min(540, W), panel_top + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov2, 0.5, frame, 0.5, 0)

        if not detected:
            sc = (180, 180, 180)
        elif score >= 80:
            sc = (90, 230, 120)
        elif score >= 60:
            sc = (255, 210, 70)
        else:
            sc = (255, 90, 90)

        if self.font_big is None or not PIL_OK:
            cv2.putText(frame, f"Q{q_no}/{q_total} Score:{score}", (14, panel_top + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (sc[2], sc[1], sc[0]), 2)
            return frame

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        self._draw_wrapped(d, f"Q{q_no}/{q_total}. " + question, 16, 12, W - 220, self.font, (255, 230, 130))
        d.text((W - 200, 16), "답변 끝나면 q", font=self.font_tip, fill=(180, 180, 180))

        y0 = panel_top + 8
        d.text((14, y0), f"종합 점수 {score}", font=self.font_big, fill=sc)
        ec = {"밝음": (90, 230, 120), "무표정": (255, 210, 70),
              "어색": (255, 170, 60), "굳음": (255, 120, 120)}.get(expression, (200, 200, 200))
        d.text((230, y0 + 4), f"표정: {expression}", font=self.font, fill=ec)

        if not shown:
            d.text((14, y0 + 40), "OK 자세·표정 모두 안정적입니다", font=self.font, fill=(90, 230, 120))
        else:
            y = y0 + 40
            for key in shown:
                d.text((14, y), "● " + ISSUE_LABELS.get(key, key), font=self.font, fill=(255, 110, 110))
                d.text((34, y + 22), "→ " + SHORT_TIPS.get(key, ""), font=self.font_tip, fill=(150, 220, 255))
                y += block_h
            if hidden_n > 0:
                d.text((14, y), f"… 외 {hidden_n}개 더", font=self.font_tip, fill=(190, 190, 190))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ════════════════════════════════════════════════════
    #  오디오 콜백
    # ════════════════════════════════════════════════════
    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_q.put(indata.copy())

    # ════════════════════════════════════════════════════
    #  한 문항 진행 (카메라는 외부에서 넘겨받아 공유)
    #  반환: (문항결과 dict, aborted)  aborted=True 면 ESC로 세션 중단
    # ════════════════════════════════════════════════════
    def run_one_question(self, cap, question, q_no, q_total, is_followup=False):
        self._reset_question_state()
        label = f"꼬리질문 (Q{q_no} 관련)" if is_followup else f"질문 {q_no}/{q_total}"
        print("\n" + "=" * 52)
        print(f"  [{label}]  {question}")
        print("=" * 52)
        print(f"[안내] {COUNTDOWN_SEC}초 후 답변. 끝나면 'q', 전체 중단은 ESC.\n")

        prev_t = time.time()
        start_t = time.time()
        recording = False
        stream = None
        aborted = False
        finished = False

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[경고] 프레임을 읽지 못했습니다.")
                    break
                fh, fw = frame.shape[:2]
                elapsed = time.time() - start_t
                in_countdown = elapsed < COUNTDOWN_SEC

                try:
                    res = self.model(frame, verbose=False)[0]
                    annotated = res.plot()
                    pts, cfs = self._extract_keypoints(res)
                except Exception as e:
                    print(f"[경고] 자세 추론 오류: {e}")
                    annotated, pts, cfs = frame.copy(), None, None
                try:
                    face = self.analyze_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fw, fh)
                except Exception as e:
                    print(f"[경고] 표정 분석 오류: {e}")
                    face = {"detected": False, "expression": "-", "issues": []}
                posture = self.analyze_posture(pts, cfs, fw)
                all_issues = posture["issues"] + face["issues"]
                detected = posture["detected"] or face["detected"]
                score = self._score_from_issues(len(all_issues)) if detected else 0

                if in_countdown:
                    sec_left = int(COUNTDOWN_SEC - elapsed) + 1
                    annotated = self._draw_countdown(annotated, sec_left, question, q_no, q_total)
                else:
                    if not recording:
                        print("[안내] 답변 녹음을 시작합니다. (끝나면 'q')\n")
                        try:
                            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                                    dtype="float32", callback=self._audio_callback)
                            stream.start()
                            recording = True
                        except Exception as e:
                            print(f"[경고] 마이크 녹음 시작 실패: {e}  (자세·표정만 진행)")
                            recording = True
                    while not self._audio_q.empty():
                        self._audio_chunks.append(self._audio_q.get())
                    if detected:
                        self.frame_count += 1
                        self.score_sum += score
                        for k in all_issues:
                            self.issue_counts[k] += 1
                        if face["detected"]:
                            self.expr_counts[face["expression"]] += 1
                    annotated = self.draw_feedback(annotated, posture, face, score, question, q_no, q_total)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                cv2.putText(annotated, f"FPS {fps:4.1f}", (annotated.shape[1] - 110, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("AI Interview Coach - Mock Interview", annotated)
                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    finished = True
                    break
                if key == 27:  # ESC
                    aborted = True
                    print("\n[안내] ESC: 세션을 중단합니다.")
                    break
        except KeyboardInterrupt:
            aborted = True
            print("\n[안내] 사용자 중단(Ctrl+C).")
        finally:
            if stream is not None:
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
            while not self._audio_q.empty():
                self._audio_chunks.append(self._audio_q.get())

        # 답변을 끝낸(=q) 경우에만 변환·평가. ESC면 건너뜀.
        result = {
            "question": question,
            "posture_score": (self.score_sum / self.frame_count) if self.frame_count else None,
            "frame_count": self.frame_count,
            "blink": self._blink_count,
            "expr": dict(self.expr_counts),
            "issues": dict(self.issue_counts),
            "answer_text": "",
            "content_score": None,
            "content_detail": None,
        }
        if finished:
            answer_text = self._finish_audio_and_transcribe()
            result["answer_text"] = answer_text
            if answer_text and EVALUATOR_AVAILABLE:
                detail = self._evaluate_content(question, answer_text)
                if detail is not None:
                    result["content_score"] = detail.overall_score
                    result["content_detail"] = detail
        return result, aborted

    def _get_whisper(self):
        if self._whisper_model is None:
            print(f"\n● Whisper 모델 로딩 중... ('{WHISPER_MODEL}', 첫 실행이면 다운로드)")
            self._whisper_model = whisper.load_model(WHISPER_MODEL)
        return self._whisper_model

    def _finish_audio_and_transcribe(self):
        if not self._audio_chunks:
            print("\n[경고] 녹음된 오디오가 없습니다. (마이크 확인) 내용 평가는 건너뜁니다.")
            return ""
        audio = np.concatenate(self._audio_chunks, axis=0).flatten()
        if float(np.abs(audio).mean()) < 0.001:
            print("\n[경고] 녹음 소리가 거의 없습니다. 마이크 입력을 확인하세요.")
        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        wav_path = os.path.join(tempfile.gettempdir(), "mock_interview_answer.wav")
        wav_write(wav_path, SAMPLE_RATE, audio_int16)
        print("● 음성을 글자로 변환 중...")
        try:
            # initial_prompt: '면접 답변'이라는 맥락 + 한글 예시를 주면 한국어 인식률이 올라감
            result = self._get_whisper().transcribe(
                wav_path,
                language="ko",
                fp16=False,
                initial_prompt="다음은 한국어 취업 면접 답변입니다. 존댓말로 또박또박 말합니다.",
                temperature=0.0,   # 무작위성 제거 → 더 안정적인 인식
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[경고] 음성 변환 실패: {e}")
            return ""

    def _evaluate_content(self, question, answer_text):
        print("● 답변 내용 평가 중 (Ollama)...")
        try:
            evaluator = ContentEvaluator()
            return evaluator.evaluate(question, answer_text, self.job_role)
        except Exception as e:
            print(f"[경고] 내용 평가 실패: {e}")
            print("   · Ollama 실행 / 'ollama pull qwen2.5:7b' 확인하세요.")
            return None

    # ════════════════════════════════════════════════════
    #  세션 전체 진행
    # ════════════════════════════════════════════════════
    def run_session(self, questions, level="중"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[에러] 웹캠을 열 수 없습니다.")
            return []

        results = []
        total = len(questions)

        # ★ 상 난이도일 때만 꼬리질문 사용. 자기소개(1번) 제외, 2~끝 중 2~3개 랜덤 선정.
        followup_targets = set()
        if level == "상" and EVALUATOR_AVAILABLE and total >= 2:
            candidates = list(range(2, total + 1))   # 2번부터
            k = min(len(candidates), random.choice([2, 3]))
            followup_targets = set(random.sample(candidates, k))
            print(f"[안내] 심화(상) 모드: {sorted(followup_targets)}번 질문에 꼬리질문이 따라붙습니다.\n")

        try:
            for i, q in enumerate(questions, start=1):
                res, aborted = self.run_one_question(cap, q, i, total)
                results.append(res)
                self._print_question_result(res, i, total)
                if aborted:
                    print("\n[안내] 세션이 중단되었습니다. 여기까지의 결과로 리포트를 만듭니다.")
                    break

                # ★ 이 문항이 꼬리질문 대상이고, 답변이 인식됐으면 꼬리질문 진행
                if i in followup_targets and res.get("answer_text"):
                    fu_q = self._make_followup(q, res["answer_text"])
                    if fu_q:
                        print(f"\n[꼬리질문] {fu_q}\n")
                        fu_res, fu_aborted = self.run_one_question(
                            cap, fu_q, i, total, is_followup=True)
                        fu_res["is_followup"] = True
                        fu_res["parent_q"] = i
                        results.append(fu_res)
                        self._print_question_result(fu_res, f"{i}-꼬리", total)
                        if fu_aborted:
                            print("\n[안내] 세션이 중단되었습니다.")
                            break

                # 마지막 문항이 아니면 전환 화면
                if i < total:
                    self._show_transition(cap, i + 1, total)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return results

    def _make_followup(self, question, answer_text):
        """답변을 보고 Ollama 로 꼬리질문 1개 생성. 실패하면 None."""
        if not EVALUATOR_AVAILABLE:
            return None
        print("\n● 답변을 바탕으로 꼬리질문을 생성하는 중...")
        try:
            import ollama
            prompt = f"""당신은 깐깐한 면접관입니다. 지원자의 답변을 듣고, 그 내용을 더 깊이 파고드는 꼬리질문 1개를 만드세요.

[원래 질문]
{question}

[지원자 답변]
{answer_text}

규칙:
- 한국어로만, 질문 1개만 출력합니다.
- 답변에서 구체적으로 언급된 내용을 파고드는 질문으로 만듭니다.
- 따옴표나 번호 없이 질문 문장만 출력하세요."""
            resp = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "당신은 한국어로만 말하는 면접관입니다. 질문 한 문장만 답하세요."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.6},
            )
            text = resp["message"]["content"].strip()
            # 혹시 여러 줄/따옴표가 오면 첫 문장만 깔끔히
            text = text.replace("\n", " ").strip().strip('"').strip("'")
            return text if text else None
        except Exception as e:
            print(f"[안내] 꼬리질문 생성 실패 → 건너뜁니다. (사유: {e})")
            return None

    def _show_transition(self, cap, next_no, total):
        start = time.time()
        while time.time() - start < TRANSITION_SEC:
            ok, frame = cap.read()
            if not ok:
                break
            sec_left = int(TRANSITION_SEC - (time.time() - start)) + 1
            frame = self._draw_transition(frame, sec_left, next_no, total)
            cv2.imshow("AI Interview Coach - Mock Interview", frame)
            if (cv2.waitKey(5) & 0xFF) == 27:
                break

    def _print_question_result(self, res, q_no, total):
        """각 문항 직후 간단 결과 출력."""
        print("\n" + "-" * 52)
        print(f"  [문항 {q_no}/{total} 결과]")
        if res["posture_score"] is not None:
            print(f"   자세·표정: {res['posture_score']:.1f}점")
        if res["answer_text"]:
            print(f"   답변: \"{res['answer_text']}\"")
        if res["content_score"] is not None:
            print(f"   내용: {res['content_score']:.1f}점")
        print("-" * 52)

    # ════════════════════════════════════════════════════
    #  종합 리포트
    # ════════════════════════════════════════════════════
    def print_final_report(self, results):
        print("\n" + "█" * 56)
        print("  모의면접 종합 리포트")
        print("█" * 56)
        if not results:
            print("  진행된 문항이 없습니다.")
            print("█" * 56)
            return None, None

        posture_scores = [r["posture_score"] for r in results if r["posture_score"] is not None]
        content_scores = [r["content_score"] for r in results if r["content_score"] is not None]
        posture_avg = sum(posture_scores) / len(posture_scores) if posture_scores else None
        content_avg = sum(content_scores) / len(content_scores) if content_scores else None

        # 문항별 표 (꼬리질문은 들여쓰기 + 표시로 구분)
        print("\n[문항별 점수]")
        print("-" * 56)
        main_no = 0
        for r in results:
            ps = f"{r['posture_score']:.0f}" if r["posture_score"] is not None else "-"
            cs = f"{r['content_score']:.0f}" if r["content_score"] is not None else "-"
            qshort = r["question"][:24] + ("…" if len(r["question"]) > 24 else "")
            if r.get("is_followup"):
                print(f"   └ 꼬리질문  자세 {ps:>3} / 내용 {cs:>3}   {qshort}")
            else:
                main_no += 1
                print(f"  Q{main_no}. 자세 {ps:>3} / 내용 {cs:>3}   {qshort}")

        # 평균
        print("\n[전체 평균]")
        print("-" * 56)
        if posture_avg is not None:
            print(f"  자세·표정 평균: {posture_avg:.1f}점 (등급 {self._to_grade(posture_avg)})")
        if content_avg is not None:
            print(f"  답변 내용 평균: {content_avg:.1f}점 (등급 {self._to_grade(content_avg)})")

        # 강·약 문항(내용 기준, 없으면 자세 기준)
        key = "content_score" if content_scores else "posture_score"
        scored = [(i + 1, r) for i, r in enumerate(results) if r[key] is not None]
        if len(scored) >= 2:
            best = max(scored, key=lambda x: x[1][key])
            worst = min(scored, key=lambda x: x[1][key])
            label = "내용" if key == "content_score" else "자세"
            print(f"\n  가장 잘한 문항: Q{best[0]} ({label} {best[1][key]:.0f}점)")
            print(f"  보완 필요 문항: Q{worst[0]} ({label} {worst[1][key]:.0f}점)")

        # 자세·표정 공통 약점(전체 문항 합산)
        total_issue = defaultdict(int)
        total_frames = 0
        for r in results:
            total_frames += r["frame_count"]
            for k, v in r["issues"].items():
                total_issue[k] += v
        if total_issue and total_frames:
            print("\n[자주 나타난 자세·표정 문제 (전체)]")
            print("-" * 56)
            for k, v in sorted(total_issue.items(), key=lambda x: -x[1])[:5]:
                print(f"  - {ISSUE_LABELS.get(k,k)} (전체의 {v/total_frames*100:.0f}%)")
                print(f"    → {ISSUE_TIPS.get(k,'-')}")

        # 각 문항 상세(내용 평가 강점/개선/모범답안)
        print("\n[문항별 상세]")
        print("=" * 56)
        for i, r in enumerate(results, start=1):
            print(f"\n● Q{i}. {r['question']}")
            if r["answer_text"]:
                print(f"   답변: \"{r['answer_text']}\"")
            else:
                print("   답변: (인식된 내용 없음)")
            d = r["content_detail"]
            if d is not None:
                print(f"   내용 {d.overall_score:.0f}점 (등급 {d.grade})")
                for k, v in d.scores.items():
                    print(f"     · {k}: {v}점 — {d.reasons.get(k,'')}")
                if d.improvements:
                    print("     [개선점]")
                    for imp in d.improvements:
                        print(f"       - {imp}")
                if d.model_answer:
                    print(f"     [모범답안] {d.model_answer}")
        print("█" * 56)
        return posture_avg, content_avg

    def save_growth(self, posture_avg, content_avg):
        if not GROWTH_AVAILABLE:
            print("\n[성장추적] growth_tracker.py 가 없어 저장을 건너뜁니다.")
            return
        if posture_avg is None and content_avg is None:
            return
        print("\n[성장추적] 이번 세션 평균을 저장합니다.")
        try:
            growth_tracker.add_record(
                posture_score=posture_avg if posture_avg is not None else 0.0,
                content_score=content_avg,
                question=f"{NUM_QUESTIONS}문항 세션",
            )
            path = growth_tracker.make_graph()
            if path:
                print(f"  성장 곡선 갱신됨 → {path}")
        except Exception as e:
            print(f"  [성장추적 실패] {e}")

    @staticmethod
    def _to_grade(s):
        if s >= 90: return "A"
        if s >= 80: return "B"
        if s >= 70: return "C"
        if s >= 60: return "D"
        return "F"

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass


def main():
    print("=" * 56)
    print(f"  AI 모의면접  ({NUM_QUESTIONS}문항 연속 · 자세·표정 + 음성 + 내용평가)")
    print("=" * 56)
    if not EVALUATOR_AVAILABLE:
        print("[참고] content_evaluator.py 를 못 불러와 내용 평가는 건너뜁니다.")
        print(f"        (사유: {_EVALUATOR_IMPORT_ERR})\n")

    # ★ 지원 직무 선택 (목록에서 번호로, 0번은 직접 입력)
    job_role = "일반 직무"
    selected_group = None   # 메뉴로 고른 직군명(매칭 우회용)
    if QBANK_AVAILABLE:
        menu = question_bank.get_job_menu()
        print("\n지원할 직무 분야를 선택하세요. (입력한 직무 기준으로 질문·평가가 맞춰집니다)")
        for i, (name, example) in enumerate(menu, start=1):
            print(f"  {i:2d}. {name:<12s}  ({example})")
        print("   0. 기타 (직접 입력)")
        try:
            sel = input("▶ 번호 선택 (그냥 Enter 시 1번): ").strip()
        except (EOFError, KeyboardInterrupt):
            sel = ""
        if sel == "0":
            try:
                typed = input("  지원 직무를 직접 입력하세요: ").strip()
            except (EOFError, KeyboardInterrupt):
                typed = ""
            job_role = typed if typed else "일반 직무"
            # 직접 입력은 selected_group 없음 → 키워드 매칭/AI 생성에 맡김
        elif sel == "":
            selected_group = menu[0][0]
            job_role = menu[0][0]
        elif sel.isdigit() and 1 <= int(sel) <= len(menu):
            selected_group = menu[int(sel) - 1][0]
            job_role = selected_group
        else:
            print("  잘못된 입력입니다. 첫 번째 직군으로 진행합니다.")
            selected_group = menu[0][0]
            job_role = menu[0][0]
    else:
        # question_bank 없을 때만 자유 입력
        print("\n먼저 지원할 직무를 알려주세요.")
        try:
            job_input = input("▶ 지원 직무 (그냥 Enter 시 '일반 직무'): ").strip()
        except (EOFError, KeyboardInterrupt):
            job_input = ""
        job_role = job_input if job_input else "일반 직무"
    print(f"[안내] '{job_role}' 기준으로 평가합니다.\n")

    # ★ 면접 난이도 입력받기 (1.하 / 2.중 / 3.상, 그냥 Enter 시 중)
    print("면접 난이도를 골라주세요. 자기소개 다음 질문들의 수준이 달라집니다.")
    print("  1. 하 (기초)  — 부담 없는 기본·동기 질문")
    print("  2. 중 (실무)  — 실무 상황·경험 질문  [기본]")
    print("  3. 상 (심화)  — 전공 지식·압박·심화 질문")
    try:
        lv_input = input("▶ 난이도 (1/2/3, 그냥 Enter 시 중): ").strip()
    except (EOFError, KeyboardInterrupt):
        lv_input = ""
    level = {"1": "하", "2": "중", "3": "상"}.get(lv_input, "중")
    print(f"[안내] 난이도 '{level}'(으)로 진행합니다.\n")

    # ★ 자소서 입력 (선택): 자소서 기반 맞춤 질문 생성
    resume_text = ""
    if QBANK_AVAILABLE:
        print("자기소개서를 활용하면, 자소서 내용을 바탕으로 맞춤 질문을 만들어 드립니다.")
        print("  1. 파일 경로 입력 (.txt 또는 .docx)")
        print("  2. 직접 붙여넣기")
        print("  3. 사용 안 함 (직무·난이도 질문으로 진행)")
        try:
            rmode = input("▶ 선택 (1/2/3, 그냥 Enter 시 3): ").strip()
        except (EOFError, KeyboardInterrupt):
            rmode = "3"
        if rmode == "1":
            try:
                rpath = input("  자소서 파일 경로: ").strip()
            except (EOFError, KeyboardInterrupt):
                rpath = ""
            resume_text = question_bank.read_resume(rpath)
        elif rmode == "2":
            print("  자소서 내용을 붙여넣고, 다 넣은 뒤 빈 줄에서 Enter를 한 번 더 누르세요:")
            lines = []
            try:
                while True:
                    line = input()
                    if line == "" and lines:   # 빈 줄 + 이미 내용 있음 → 종료
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                pass
            resume_text = "\n".join(lines).strip()
        if resume_text:
            print(f"[안내] 자소서를 받았습니다. ({len(resume_text)}자) 이 내용으로 질문을 만듭니다.\n")
        else:
            print("[안내] 자소서 없이 진행합니다.\n")

    # ★ 질문 구성: 자소서가 있으면 자소서 기반, 없으면 직무·난이도
    if QBANK_AVAILABLE:
        if resume_text:
            questions = question_bank.build_questions_from_resume(
                resume_text, job_role=job_role, level=level, n=NUM_QUESTIONS)
        else:
            questions = question_bank.build_questions(
                job_role=job_role, level=level, n=NUM_QUESTIONS, group=selected_group)
        n = len(questions)
    else:
        n = min(NUM_QUESTIONS, len(QUESTIONS))
        questions = random.sample(QUESTIONS, n)

    print(f"\n[안내] 총 {n}문항을 진행합니다. 각 문항: 준비 {COUNTDOWN_SEC}초 → 답변 → 'q'.")
    print("[안내] 중간에 그만두려면 카메라 창에서 ESC 를 누르세요.\n")

    interview = MockInterview(job_role=job_role)
    try:
        results = interview.run_session(questions, level=level)
        posture_avg, content_avg = interview.print_final_report(results)
        interview.save_growth(posture_avg, content_avg)

        # ★ HTML 리포트 생성 + 브라우저로 열기
        if REPORT_AVAILABLE and results:
            print("\n[리포트] HTML 결과 리포트를 생성합니다...")
            try:
                # 성장 곡선 이미지가 있으면 리포트에 함께 삽입
                graph_path = None
                if GROWTH_AVAILABLE:
                    g = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "growth_curve.png")
                    if os.path.exists(g):
                        graph_path = g
                path = report_generator.generate(
                    results, posture_avg, content_avg,
                    graph_path=graph_path, job_role=interview.job_role)
                if path:
                    report_generator.open_in_browser(path)
                    print(f"  리포트가 브라우저에서 열립니다 → {path}")
            except Exception as e:
                print(f"  [리포트 생성 실패] {e}")
        elif not REPORT_AVAILABLE:
            print("\n[리포트] report_generator.py 를 같은 폴더에 두면 HTML 리포트가 생성됩니다.")
    finally:
        interview.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[에러] 실행 실패: {e}")
        sys.exit(1)
