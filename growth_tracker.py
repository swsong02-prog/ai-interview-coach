"""
growth_tracker.py
────────────────────────────────────────────────────────
회차별 성장 추적 (점수 저장 + 학습 곡선 그래프)

핵심 기능
  1. 모의면접 점수를 회차별로 JSON 파일에 저장 (add_record)
  2. 쌓인 기록을 예쁜 성장 곡선 PNG 로 시각화 (make_graph)
  3. mock_interview.py 가 면접 종료 시 자동으로 점수를 저장하도록 연결

저장 위치
  - interview_history.json : 회차별 기록(텍스트, 사람이 읽을 수 있음)
  - growth_curve.png       : 성장 곡선 이미지

설치
  pip install matplotlib

사용
  · 단독 실행(그래프 보기) :  python growth_tracker.py
  · mock_interview.py 안에서 자동 호출됨 (직접 부를 필요 없음)
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import json
from datetime import datetime

# matplotlib 는 그래프용. 없으면 저장은 되지만 그래프만 건너뜀.
try:
    import matplotlib
    matplotlib.use("Agg")   # 창 없이 파일로 저장(서버/터미널 환경 안전)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ── 설정 ────────────────────────────────────────────────
# 이 파일이 있는 폴더 기준으로 저장 → 어디서 실행해도 같은 위치에 쌓임
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "interview_history.json")
GRAPH_PATH = os.path.join(BASE_DIR, "growth_curve.png")


def _setup_korean_font():
    """matplotlib 한글 깨짐(□□□) 방지: 맑은 고딕 등 한글 폰트 지정."""
    if not MPL_OK:
        return
    for path in ["C:/Windows/Fonts/malgun.ttf",
                 "C:/Windows/Fonts/malgunbd.ttf",
                 "C:/Windows/Fonts/gulim.ttc"]:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지


# ════════════════════════════════════════════════════════
#  저장
# ════════════════════════════════════════════════════════
def load_history() -> list:
    """기존 기록 불러오기. 없으면 빈 리스트."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[경고] 기록 파일을 읽지 못했습니다: {e}  (새로 시작합니다)")
        return []


def add_record(posture_score: float,
               content_score: float | None = None,
               question: str = "",
               extra: dict | None = None) -> dict:
    """
    한 회차 점수를 기록에 추가하고 저장.
    :param posture_score: 자세·표정 종합 점수 (0~100)
    :param content_score: 답변 내용 종합 점수 (없으면 None)
    :param question: 그 회차 질문(선택)
    :param extra: 추가로 남기고 싶은 값(선택, dict)
    :return: 방금 저장한 기록 dict
    """
    history = load_history()
    record = {
        "session": len(history) + 1,                       # 회차 번호
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),  # 날짜·시각
        "posture_score": round(float(posture_score), 1),
        "content_score": (round(float(content_score), 1)
                          if content_score is not None else None),
        "question": question,
    }
    if extra:
        record.update(extra)
    history.append(record)

    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[경고] 기록 저장 실패: {e}")

    print(f"[성장추적] {record['session']}회차 기록 저장됨 "
          f"(자세 {record['posture_score']}점"
          + (f", 내용 {record['content_score']}점)" if record['content_score'] is not None else ")"))
    return record


# ════════════════════════════════════════════════════════
#  그래프
# ════════════════════════════════════════════════════════
def make_graph(save_path: str = GRAPH_PATH) -> str | None:
    """쌓인 기록으로 성장 곡선 PNG 생성. 경로 반환(실패 시 None)."""
    if not MPL_OK:
        print("[안내] matplotlib 가 없어 그래프를 만들지 못합니다.  pip install matplotlib")
        return None

    history = load_history()
    if len(history) == 0:
        print("[안내] 저장된 기록이 없습니다. 모의면접을 먼저 진행하세요.")
        return None

    _setup_korean_font()

    sessions = [r["session"] for r in history]
    posture = [r.get("posture_score") for r in history]
    content = [r.get("content_score") for r in history]
    has_content = any(c is not None for c in content)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # 자세·표정 곡선
    ax.plot(sessions, posture, marker="o", linewidth=2.5, markersize=8,
            color="#5ce6a0", label="자세·표정 점수")
    for x, y in zip(sessions, posture):
        if y is not None:
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", color="#5ce6a0", fontsize=9)

    # 내용 곡선 (있을 때만)
    if has_content:
        cx = [s for s, c in zip(sessions, content) if c is not None]
        cy = [c for c in content if c is not None]
        ax.plot(cx, cy, marker="s", linewidth=2.5, markersize=8,
                color="#78b6ff", label="답변 내용 점수")
        for x, y in zip(cx, cy):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, -16), ha="center", color="#78b6ff", fontsize=9)

    # 꾸미기
    ax.set_title("모의면접 성장 곡선", color="white", fontsize=16, pad=15)
    ax.set_xlabel("회차", color="#cccccc", fontsize=11)
    ax.set_ylabel("점수", color="#cccccc", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(sessions)
    ax.grid(True, alpha=0.15, color="white")
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    legend = ax.legend(loc="lower right", framealpha=0.2)
    for text in legend.get_texts():
        text.set_color("white")

    # 첫 회차 대비 향상 표시(자세 기준)
    valid_p = [p for p in posture if p is not None]
    if len(valid_p) >= 2:
        diff = valid_p[-1] - valid_p[0]
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "−")
        color = "#5ce6a0" if diff > 0 else ("#ff7070" if diff < 0 else "#cccccc")
        ax.text(0.02, 0.95, f"첫 회차 대비 자세 {sign} {abs(diff):.0f}점",
                transform=ax.transAxes, color=color, fontsize=11, va="top")

    try:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[성장추적] 성장 곡선 저장됨 → {save_path}")
        return save_path
    except Exception as e:
        print(f"[경고] 그래프 저장 실패: {e}")
        return None


def print_text_summary():
    """터미널에 회차별 점수 간단 요약(그래프 못 볼 때 보조)."""
    history = load_history()
    if not history:
        print("저장된 기록이 없습니다.")
        return
    print("\n" + "=" * 52)
    print("  회차별 기록")
    print("=" * 52)
    for r in history:
        c = f", 내용 {r['content_score']}점" if r.get("content_score") is not None else ""
        print(f"  {r['session']}회차 ({r['date']}) — 자세 {r['posture_score']}점{c}")
    print("=" * 52)


# ── 단독 실행: 지금까지 기록으로 그래프 만들기 ─────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  성장 추적 — 그래프 생성")
    print("=" * 52)
    print_text_summary()
    path = make_graph()
    if path:
        print(f"\n생성된 그래프 파일을 열어보세요: {path}")
