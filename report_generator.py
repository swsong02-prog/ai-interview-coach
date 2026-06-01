"""
report_generator.py
────────────────────────────────────────────────────────
모의면접 종합 결과를 예쁜 HTML 리포트 파일로 저장.

핵심 기능
  1. 5문항 종합 결과(점수·답변·평가·모범답안) → HTML 한 장으로
  2. 성장 곡선 이미지(growth_curve.png)가 있으면 함께 삽입
  3. 저장 후 브라우저에서 자동으로 열기(open_in_browser)

특징
  - 외부 라이브러리 불필요(순수 파이썬 + HTML/CSS)
  - 한글 깨짐 없음(UTF-8)
  - mock_interview.py 가 면접 종료 시 자동 호출

사용
  · mock_interview.py 안에서 자동 호출됨
  · 단독 테스트 :  python report_generator.py   (샘플 리포트 생성)
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import base64
import webbrowser
from datetime import datetime
from html import escape

# 이 파일 폴더 기준으로 저장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _grade_color(score):
    """점수 → 색상(등급별)."""
    if score is None:
        return "#9aa0a6"
    if score >= 90:
        return "#34d399"   # A 초록
    if score >= 80:
        return "#60d394"
    if score >= 70:
        return "#fbbf24"   # C 노랑
    if score >= 60:
        return "#fb923c"
    return "#f87171"       # F 빨강


def _to_grade(score):
    if score is None:
        return "-"
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _img_base64(path):
    """이미지를 base64로 인코딩해 HTML에 직접 박아넣기(파일 의존 X)."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


# 자세·표정 문제 라벨/코칭 (mock_interview 와 동일 키)
ISSUE_LABELS = {
    "shoulder_tilt": "어깨가 한쪽으로 기울어짐",
    "head_tilt": "고개가 기울어짐",
    "not_facing": "정면을 바라보지 않음",
    "off_center": "화면 중앙에서 벗어남",
    "too_close": "카메라와 너무 가까움",
    "too_far": "카메라와 너무 멂",
    "head_down": "고개가 떨어짐",
    "stiff_face": "표정이 굳어 있음",
    "gaze_away": "시선이 정면을 벗어남",
}
ISSUE_TIPS = {
    "shoulder_tilt": "양쪽 어깨 높이를 맞추고 균형 있게 앉으세요.",
    "head_tilt": "고개를 똑바로 세워 화면과 수평을 맞추세요.",
    "not_facing": "면접관(카메라)을 정면으로 바라보세요.",
    "off_center": "상반신이 화면 중앙에 오도록 위치를 잡으세요.",
    "too_close": "카메라에서 조금 떨어져 상반신이 다 보이게 하세요.",
    "too_far": "카메라에 조금 더 가까이 앉으세요.",
    "head_down": "고개를 들고 시선을 정면으로 향하세요.",
    "stiff_face": "자연스러운 미소가 호감을 줍니다. 너무 굳지 않게 연습하세요.",
    "gaze_away": "답변 중에도 카메라(면접관)에 시선을 두세요.",
}


def generate(results, posture_avg, content_avg,
             graph_path=None, save_path=None, job_role="일반 직무") -> str | None:
    """
    종합 결과 → HTML 파일 저장. 저장 경로 반환(실패 시 None).
    :param results: mock_interview 의 문항별 결과 리스트
    :param posture_avg: 자세·표정 평균
    :param content_avg: 내용 평균
    :param graph_path: 성장곡선 PNG 경로(있으면 삽입)
    :param save_path: 저장 경로(None이면 날짜 기반 자동)
    :param job_role: 지원 직무(헤더에 표시)
    """
    now = datetime.now()
    if save_path is None:
        fname = "report_" + now.strftime("%Y%m%d_%H%M%S") + ".html"
        save_path = os.path.join(BASE_DIR, fname)

    # ── 자세·표정 공통 약점 합산 ──
    total_issue = {}
    total_frames = 0
    for r in results:
        total_frames += r.get("frame_count", 0)
        for k, v in (r.get("issues", {}) or {}).items():
            total_issue[k] = total_issue.get(k, 0) + v

    # ── 문항별 카드 HTML ──
    cards = []
    for i, r in enumerate(results, start=1):
        q = escape(r.get("question", ""))
        ans = escape(r.get("answer_text", "") or "(인식된 답변 없음)")
        cs = r.get("content_score")
        ps = r.get("posture_score")
        detail = r.get("content_detail")

        # 점수 배지
        badge_p = (f'<span class="badge" style="background:{_grade_color(ps)}">'
                   f'자세 {ps:.0f}</span>' if ps is not None else "")
        badge_c = (f'<span class="badge" style="background:{_grade_color(cs)}">'
                   f'내용 {cs:.0f}</span>' if cs is not None else "")

        # 내용 평가 상세
        detail_html = ""
        if detail is not None:
            rows = ""
            for k, v in detail.scores.items():
                reason = escape(detail.reasons.get(k, ""))
                rows += (f'<tr><td class="crit">{escape(k)}</td>'
                         f'<td class="num" style="color:{_grade_color(v)}">{v}</td>'
                         f'<td class="reason">{reason}</td></tr>')
            strengths = "".join(f"<li>{escape(s)}</li>" for s in (detail.strengths or []))
            improvements = "".join(f"<li>{escape(x)}</li>" for x in (detail.improvements or []))
            model_ans = escape(detail.model_answer or "")
            detail_html = f"""
            <table class="crit-table">
              <thead><tr><th>항목</th><th>점수</th><th>이유</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            <div class="two-col">
              <div class="box good"><div class="box-title">잘한 점</div><ul>{strengths or '<li>-</li>'}</ul></div>
              <div class="box bad"><div class="box-title">개선할 점</div><ul>{improvements or '<li>-</li>'}</ul></div>
            </div>
            <div class="model-answer">
              <div class="box-title">모범 답안</div>
              <p>{model_ans or '-'}</p>
            </div>"""
        else:
            detail_html = '<p class="muted">내용 평가 결과가 없습니다.</p>'

        cards.append(f"""
        <div class="card">
          <div class="card-head">
            <span class="qno">Q{i}</span>
            <span class="qtext">{q}</span>
            <span class="badges">{badge_p}{badge_c}</span>
          </div>
          <div class="answer"><span class="label">내 답변</span> {ans}</div>
          {detail_html}
        </div>""")

    cards_html = "\n".join(cards)

    # ── 자세·표정 약점 섹션 ──
    posture_html = ""
    if total_issue and total_frames:
        items = ""
        for k, v in sorted(total_issue.items(), key=lambda x: -x[1])[:6]:
            pct = v / total_frames * 100
            items += (f'<li><b>{escape(ISSUE_LABELS.get(k,k))}</b> '
                      f'<span class="muted">(전체의 {pct:.0f}%)</span><br>'
                      f'<span class="tip">→ {escape(ISSUE_TIPS.get(k,""))}</span></li>')
        posture_html = f"""
        <div class="section">
          <h2>자세 · 표정에서 고칠 점</h2>
          <ul class="issue-list">{items}</ul>
        </div>"""

    # ── 성장 곡선 이미지 ──
    graph_html = ""
    if graph_path:
        b64 = _img_base64(graph_path)
        if b64:
            graph_html = f"""
            <div class="section">
              <h2>회차별 성장 곡선</h2>
              <img class="graph" src="{b64}" alt="성장 곡선">
            </div>"""

    # ── 종합 점수 헤더 ──
    p_disp = f"{posture_avg:.1f}" if posture_avg is not None else "-"
    c_disp = f"{content_avg:.1f}" if content_avg is not None else "-"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>모의면접 리포트 — {now.strftime("%Y-%m-%d %H:%M")}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Malgun Gothic','맑은 고딕',system-ui,sans-serif;
    background:#0f1117; color:#e8e8ec; line-height:1.6;
    padding: 32px 16px;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  header {{ text-align:center; margin-bottom: 28px; }}
  header h1 {{ font-size: 28px; font-weight: 700; }}
  header .date {{ color:#9aa0a6; margin-top: 6px; font-size: 14px; }}
  .summary {{
    display:flex; gap:16px; justify-content:center; margin: 24px 0 32px;
    flex-wrap: wrap;
  }}
  .score-card {{
    background:#1a1d27; border:1px solid #2a2e3a; border-radius:16px;
    padding: 22px 36px; text-align:center; min-width: 180px;
  }}
  .score-card .lbl {{ color:#9aa0a6; font-size:14px; margin-bottom:8px; }}
  .score-card .val {{ font-size:44px; font-weight:800; line-height:1; }}
  .score-card .grade {{ font-size:15px; margin-top:8px; color:#cbd5e1; }}
  .section {{
    background:#161922; border:1px solid #262a36; border-radius:14px;
    padding: 22px 24px; margin-bottom: 20px;
  }}
  .section h2 {{ font-size:19px; margin-bottom:14px; color:#f1f5f9; }}
  .card {{
    background:#161922; border:1px solid #262a36; border-radius:14px;
    padding: 20px 22px; margin-bottom: 18px;
  }}
  .card-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
  .qno {{ background:#3b82f6; color:#fff; font-weight:700; padding:3px 11px; border-radius:8px; font-size:14px; }}
  .qtext {{ font-weight:600; font-size:16px; flex:1; min-width:200px; }}
  .badges {{ display:flex; gap:6px; }}
  .badge {{ color:#0b0e14; font-weight:700; font-size:13px; padding:3px 10px; border-radius:20px; }}
  .answer {{ background:#0f1117; border-radius:10px; padding:12px 14px; margin-bottom:14px; font-size:15px; }}
  .answer .label {{ color:#60a5fa; font-weight:700; margin-right:6px; font-size:13px; }}
  .label {{ }}
  .crit-table {{ width:100%; border-collapse:collapse; margin-bottom:14px; font-size:14px; }}
  .crit-table th {{ text-align:left; color:#9aa0a6; font-weight:600; padding:6px 8px; border-bottom:1px solid #2a2e3a; }}
  .crit-table td {{ padding:8px; border-bottom:1px solid #1f2430; vertical-align:top; }}
  .crit-table .crit {{ font-weight:600; width:84px; }}
  .crit-table .num {{ font-weight:800; width:48px; text-align:center; }}
  .crit-table .reason {{ color:#cbd5e1; }}
  .two-col {{ display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap; }}
  .box {{ flex:1; min-width:200px; border-radius:10px; padding:12px 14px; }}
  .box.good {{ background:rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.25); }}
  .box.bad {{ background:rgba(248,113,113,.08); border:1px solid rgba(248,113,113,.25); }}
  .box-title {{ font-weight:700; margin-bottom:6px; font-size:14px; }}
  .box ul {{ padding-left:18px; font-size:14px; }}
  .model-answer {{ background:rgba(96,165,250,.08); border:1px solid rgba(96,165,250,.25); border-radius:10px; padding:12px 14px; }}
  .model-answer p {{ font-size:15px; color:#e8eef7; }}
  .issue-list {{ list-style:none; }}
  .issue-list li {{ padding:10px 0; border-bottom:1px solid #1f2430; }}
  .issue-list li:last-child {{ border-bottom:none; }}
  .tip {{ color:#7dd3fc; font-size:14px; }}
  .muted {{ color:#9aa0a6; font-size:13px; }}
  .graph {{ width:100%; border-radius:10px; background:#0f1117; }}
  footer {{ text-align:center; color:#6b7280; font-size:13px; margin-top:28px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🎯 AI 모의면접 결과 리포트</h1>
    <div class="date">{now.strftime("%Y년 %m월 %d일 %H:%M")} · 총 {len(results)}문항 · 지원 직무: {escape(job_role)}</div>
  </header>

  <div class="summary">
    <div class="score-card">
      <div class="lbl">자세 · 표정 평균</div>
      <div class="val" style="color:{_grade_color(posture_avg)}">{p_disp}</div>
      <div class="grade">등급 {_to_grade(posture_avg)}</div>
    </div>
    <div class="score-card">
      <div class="lbl">답변 내용 평균</div>
      <div class="val" style="color:{_grade_color(content_avg)}">{c_disp}</div>
      <div class="grade">등급 {_to_grade(content_avg)}</div>
    </div>
  </div>

  {graph_html}
  {posture_html}

  <div class="section">
    <h2>문항별 상세</h2>
  </div>
  {cards_html}

  <footer>AI 면접 코치 · 학생용 무료 오픈소스 프로젝트</footer>
</div>
</body>
</html>"""

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[리포트] HTML 리포트 저장됨 → {save_path}")
        return save_path
    except Exception as e:
        print(f"[경고] 리포트 저장 실패: {e}")
        return None


def open_in_browser(path):
    """저장된 HTML을 기본 브라우저로 연다."""
    if not path:
        return
    try:
        webbrowser.open("file://" + os.path.abspath(path))
    except Exception as e:
        print(f"[안내] 브라우저 자동 열기 실패: {e}")
        print(f"       직접 파일을 열어보세요: {path}")


# ── 단독 테스트: 샘플 데이터로 리포트 생성 ───────────────────
if __name__ == "__main__":
    class _D:
        def __init__(s, ov, scores, reasons, st, im, ans):
            s.overall_score = ov; s.grade = _to_grade(ov); s.scores = scores
            s.reasons = reasons; s.strengths = st; s.improvements = im; s.model_answer = ans

    sample = [
        {"question": "간단하게 자기소개를 해주세요.", "posture_score": 88.0,
         "frame_count": 200, "issues": {"head_down": 40, "stiff_face": 15},
         "answer_text": "안녕하세요, 저는 책임감이 강한 지원자입니다.",
         "content_score": 72.0,
         "content_detail": _D(72.0,
            {"논리성": 75, "구체성": 65, "직무적합도": 76},
            {"논리성": "구성이 비교적 명확합니다.", "구체성": "사례가 부족합니다.", "직무적합도": "직무 연관성이 보입니다."},
            ["자신감 있는 태도", "명확한 첫인상"], ["구체적 경험 사례 추가 필요"],
            "안녕하세요. 저는 맡은 일을 끝까지 책임지는 개발자입니다. 지난 프로젝트에서 일정이 지연됐을 때 끝까지 남아 마무리한 경험이 있습니다.")},
        {"question": "본인의 단점은 무엇인가요?", "posture_score": 80.0,
         "frame_count": 180, "issues": {"gaze_away": 30},
         "answer_text": "완벽주의 성향이 있어 때때로 시간이 오래 걸립니다.",
         "content_score": 60.0,
         "content_detail": _D(60.0,
            {"논리성": 62, "구체성": 55, "직무적합도": 63},
            {"논리성": "흐름은 자연스럽습니다.", "구체성": "극복 노력의 사례가 약합니다.", "직무적합도": "보통입니다."},
            ["솔직한 자기 인식"], ["단점 극복 노력을 구체적으로"],
            "완벽주의 성향이 있지만, 우선순위를 정해 마감을 지키는 습관으로 보완하고 있습니다.")},
    ]
    path = generate(sample, posture_avg=84.0, content_avg=66.0, graph_path=None)
    print("샘플 리포트 생성 완료:", path)
