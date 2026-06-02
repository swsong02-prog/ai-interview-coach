"""
content_evaluator.py  (Ollama 무료 버전 / 한국어 강제 + 중국어 혼입 방지)
────────────────────────────────────────────────────────
면접 답변 "내용" 평가 엔진 — 로컬 무료 AI(Ollama) 기반

핵심 기능
  1. 답변 내용 평가  : 논리성 / 구체성 / 직무적합도  (점수 + 등급)
  2. 이유 설명       : "왜 이 점수인지" 한국어로 친절하게
  3. 모범 답안 생성  : 사용자의 답변을 살린 개선 버전 자동 생성

이번 개선 (중국어 등 외국어 혼입 방지)
  - 시스템/사용자 프롬프트에 "한국어로만 답하라" 강하게 명시
  - 모범답안 길이 제한(3~4문장) → 길게 흐르다 외국어로 새는 것 방지
  - 응답 후처리: 한자/일본어 문자가 섞이면 잘라내는 안전장치

사전 준비
  1) Ollama 설치        : https://ollama.com
  2) 모델 다운로드      : ollama pull qwen2.5:7b
  3) 파이썬 라이브러리  : pip install ollama

실행
  python content_evaluator.py
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import ollama
except ImportError as e:
    raise ImportError("ollama 패키지가 필요합니다:  pip install ollama") from e


# ── 설정 ────────────────────────────────────────────────
DEFAULT_MODEL = "qwen2.5:7b"   # 컴퓨터가 버거우면 "qwen2.5:3b"
MAX_RETRY = 4                  # CUDA 오류 시 CPU 모드 재시도 여유분 포함
RETRY_DELAY = 2.0


@dataclass
class EvaluationResult:
    scores: dict
    overall_score: float
    grade: str
    reasons: dict
    strengths: list
    improvements: list
    model_answer: str
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "reasons": self.reasons,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "model_answer": self.model_answer,
        }

    def pretty_print(self) -> None:
        print("=" * 52)
        print(f"  종합 점수: {self.overall_score:.1f}점   (등급: {self.grade})")
        print("=" * 52)
        for k, v in self.scores.items():
            print(f"  • {k}: {v}점")
            print(f"      └ 이유: {self.reasons.get(k, '-')}")
        print("\n[강점]")
        for s in self.strengths:
            print(f"  + {s}")
        print("\n[개선점]")
        for i in self.improvements:
            print(f"  - {i}")
        print("\n[모범 답안]")
        print(f"  {self.model_answer}")
        print("=" * 52)


# ── 외국어(중국어/일본어) 혼입 방지용 후처리 ────────────────
# 한중일 한자(CJK), 일본어 가나 영역을 잡아냄
_CJK_PATTERN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)


def _strip_foreign(text: str) -> str:
    """
    한국어 텍스트에 중국어/일본어가 섞이면 그 지점부터 잘라냄.
    (qwen 모델이 길게 답하다 모국어로 새는 현상 대비 안전장치)
    """
    if not text:
        return text
    m = _CJK_PATTERN.search(text)
    if not m:
        return text.strip()
    # 외국어가 처음 나타난 지점 직전까지만 사용
    cut = text[:m.start()].strip()
    # 너무 많이 잘려나가면(앞부분도 거의 외국어면) 원본을 둔다
    if len(cut) < 5:
        return text.strip()
    # 끝이 어정쩡하게 잘렸으면 마지막 문장 부호까지만
    for sep in ["다.", "요.", ".", "!", "?"]:
        idx = cut.rfind(sep)
        if idx != -1:
            return cut[:idx + len(sep)].strip()
    return cut


class ContentEvaluator:
    """로컬 무료 AI(Ollama) 기반 면접 답변 '내용' 평가기"""

    # ★ 한국어 강제를 시스템 프롬프트에서 강하게 명시
    SYSTEM_PROMPT = (
        "당신은 한국 대학생의 취업 면접을 돕는 전문 면접 코치입니다. "
        "단순히 점수만 주지 말고, 왜 그렇게 평가했는지 친절하고 구체적으로 설명하세요. "
        "학생이 성장할 수 있도록 격려하되, 개선점은 명확하게 짚어 주세요.\n"
        "매우 중요한 규칙:\n"
        "1) 모든 출력은 100% 한국어로만 작성합니다. 중국어, 일본어, 영어 문장을 절대 쓰지 마세요.\n"
        "2) 반드시 지정된 JSON 형식 하나만 출력합니다. JSON 외의 설명이나 인사말을 붙이지 마세요.\n"
        "3) JSON의 모든 문자열 값도 한국어로만 작성합니다."
    )

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._check_model_ready()

    def _check_model_ready(self) -> None:
        try:
            installed = ollama.list()
            names = [m.get("model", "") for m in installed.get("models", [])]
            if not any(self.model.split(":")[0] in n for n in names):
                print(f"[경고] '{self.model}' 모델이 안 보입니다. "
                      f"터미널에서 먼저 받아주세요:  ollama pull {self.model}")
        except Exception as e:
            print("[경고] Ollama에 연결할 수 없습니다. 설치·실행 중인지 확인하세요. "
                  f"(https://ollama.com)\n        상세: {e}")

    def evaluate(self, question: str, answer: str, job_role: str = "일반 직무") -> EvaluationResult:
        if not answer or not answer.strip():
            return EvaluationResult(
                scores={"논리성": 0, "구체성": 0, "직무적합도": 0},
                overall_score=0.0, grade="F",
                reasons={"논리성": "답변이 비어 있어 평가할 수 없습니다.",
                         "구체성": "답변이 비어 있어 평가할 수 없습니다.",
                         "직무적합도": "답변이 비어 있어 평가할 수 없습니다."},
                strengths=[], improvements=["먼저 질문에 대한 답변을 작성해 주세요."],
                model_answer="(답변 입력 후 다시 시도해 주세요.)",
            )
        prompt = self._build_prompt(question, answer, job_role)
        raw = self._call_model(prompt)
        result = self._parse(raw)
        # ★ 안전장치: 명백히 부실한 답변은 모델이 후하게 줘도 점수 상한을 낮춘다
        result = self._apply_low_effort_guard(answer, result)
        return result

    # ── 동문서답·빈약한 답변 가드 ────────────────────────────
    # 모델이 격려하려다 과하게 후한 점수를 주는 경우를 코드 단에서 한 번 더 보정.
    _NON_ANSWER_PATTERNS = [
        "없어요", "없습니다", "모르겠", "모르겠어요", "글쎄요", "잘 모르",
        "패스", "스킵", "넘어갈", "생각 안", "생각이 안", "기억 안", "기억이 안",
    ]

    def _apply_low_effort_guard(self, answer: str, result: "EvaluationResult") -> "EvaluationResult":
        text = (answer or "").strip()
        # 한글/영문/숫자만 남겨 길이 측정(문장부호·공백 제외)
        core = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
        n = len(core)

        cap = None        # 점수 상한
        note = None       # 사유 메모

        low = text.replace(" ", "")
        is_non_answer = any(p.replace(" ", "") in low for p in self._NON_ANSWER_PATTERNS) and n <= 25

        if n < 6 or is_non_answer:
            # 사실상 답을 하지 않음 → 0~15
            cap, note = 15, "질문에 대한 실질적인 답변이 거의 없습니다."
        elif n < 15:
            # 매우 짧음 → 16~34 구간 상한
            cap, note = 30, "답변이 매우 짧아 내용을 평가하기 어렵습니다."

        if cap is None:
            return result  # 가드 불필요

        # 상한 적용
        new_scores = {k: min(v, cap) for k, v in result.scores.items()}
        new_overall = round(sum(new_scores.values()) / len(new_scores), 1)
        # 이유에 안내 한 줄 덧붙이기(중복 방지)
        new_reasons = dict(result.reasons)
        for k in new_reasons:
            if note not in new_reasons[k]:
                new_reasons[k] = f"{note} {new_reasons[k]}".strip()
        # 개선점에 핵심 안내 추가
        new_improvements = list(result.improvements or [])
        guide = "질문 의도에 맞춰, 구체적인 경험이나 생각을 한두 문장 이상으로 답해 보세요."
        if guide not in new_improvements:
            new_improvements.insert(0, guide)

        return EvaluationResult(
            scores=new_scores,
            overall_score=new_overall,
            grade=self._to_grade(new_overall),
            reasons=new_reasons,
            strengths=result.strengths,
            improvements=new_improvements,
            model_answer=result.model_answer,
            raw=result.raw,
        )

    def _build_prompt(self, question: str, answer: str, job_role: str) -> str:
        # ★ 한국어 강제 + 채점 기준표(루브릭) + 동문서답/빈답 가드
        return f"""[면접 질문]
{question}

[지원 직무]
{job_role}

[지원자 답변]
{answer}

위 답변을 다음 세 가지 기준으로 0~100점으로 평가하세요.
1. 논리성     : 답변이 논리적으로 구성되어 있고 일관성이 있는가
2. 구체성     : 구체적인 경험·수치·사례가 포함되어 있는가
3. 직무적합도 : 지원 직무와 연관된 역량이 드러나는가

[채점 기준표 — 반드시 이 기준을 지켜 점수를 매기세요]
- 85~100점: 질문에 정확히 답하고, 구체적 경험·근거가 풍부하며, 논리가 명확함.
- 70~84점 : 질문에 잘 답했고 흐름도 자연스러우나, 구체적 사례나 깊이가 조금 아쉬움.
- 55~69점 : 질문 의도엔 맞으나 내용이 다소 일반적이고 근거가 부족함. (평범한 답변의 기본 구간)
- 35~54점 : 답은 했으나 추상적이고 질문과 느슨하게만 연결됨.
- 16~34점 : 답변이 매우 짧거나 성의가 부족하고, 질문에 거의 답하지 못함.
- 0~15점  : 질문과 전혀 무관한 동문서답이거나, 의미 없는 말이거나, 사실상 답을 하지 않음
            (예: "없어요", "모르겠어요", "글쎄요", 질문과 상관없는 엉뚱한 말).

[중요 지침]
- 기본 태도는 '격려하는 코치'입니다. 제대로 답한 부분은 따뜻하게 인정하고, 점수는 너무 인색하지 않게 줍니다.
- 다만 위 채점 기준표는 반드시 지킵니다. 특히 질문에 답하지 못했거나 동문서답인 경우,
  격려하려는 마음 때문에 점수를 후하게 주지 마세요. 그런 답변은 정직하게 0~15점을 줍니다.
- reasons(이유)는 친절하고 구체적으로 쓰되, 부족한 점은 분명히 짚어 줍니다.
- 답변이 부실하더라도 strengths(강점)에는 격려가 될 만한 점을 한 가지는 찾아 적습니다.

[작성 규칙]
- 모든 내용은 반드시 한국어로만 작성합니다. (중국어·일본어·영어 문장 금지)
- model_answer(모범 답안)는 3~4문장 이내로 간결하게 작성합니다.
- 아래 JSON 형식 하나만 출력하고, 그 외 텍스트는 절대 붙이지 마세요.

{{
  "scores": {{"논리성": 정수, "구체성": 정수, "직무적합도": 정수}},
  "reasons": {{"논리성": "한국어 이유", "구체성": "한국어 이유", "직무적합도": "한국어 이유"}},
  "strengths": ["한국어 강점1", "한국어 강점2"],
  "improvements": ["한국어 개선점1", "한국어 개선점2"],
  "model_answer": "지원자의 답변을 살리되 더 논리적이고 구체적으로 개선한 3~4문장의 한국어 모범 답안"
}}"""

    def _call_model(self, user_prompt: str) -> dict:
        last_err = None
        use_cpu = False   # ★ CUDA 에러가 나면 True 로 바꿔 CPU 강제 모드로 재시도
        for attempt in range(1, MAX_RETRY + 1):
            try:
                # 옵션 구성. CPU 폴백 모드면 num_gpu=0 으로 GPU 사용 안 함.
                options = {"temperature": 0.4}
                if use_cpu:
                    options["num_gpu"] = 0   # GPU 레이어 0개 → 전부 CPU로 실행
                resp = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json",
                    options=options,
                )
                content = resp["message"]["content"]
                return self._safe_json(content)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # ★ CUDA/GPU 관련 에러면, 다음 시도는 CPU 모드로 전환
                if ("cuda" in msg or "gpu" in msg or "runner process has terminated" in msg) \
                        and not use_cpu:
                    use_cpu = True
                    print(f"[경고] GPU(CUDA) 오류 감지 → CPU 모드로 다시 시도합니다. ({attempt}/{MAX_RETRY})")
                    print(f"        상세: {e}")
                    # CPU 모드 재시도는 바로 진행(대기 짧게)
                    time.sleep(1.0)
                    continue
                print(f"[경고] 모델 호출 실패 ({attempt}/{MAX_RETRY}): {e}")
                if attempt < MAX_RETRY:
                    time.sleep(RETRY_DELAY)
        raise RuntimeError(
            "로컬 모델 평가에 실패했습니다. Ollama 실행 여부와 모델 설치를 확인하세요. "
            f"(원인: {last_err})"
        )

    @staticmethod
    def _safe_json(text: str) -> dict:
        if not text:
            raise ValueError("빈 응답을 받았습니다.")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)

    def _parse(self, raw: dict) -> EvaluationResult:
        scores = raw.get("scores", {}) or {}
        norm_scores = {
            "논리성": self._as_int(scores.get("논리성", 0)),
            "구체성": self._as_int(scores.get("구체성", 0)),
            "직무적합도": self._as_int(scores.get("직무적합도", 0)),
        }
        overall = sum(norm_scores.values()) / len(norm_scores)

        # ★ 외국어 혼입 후처리: 이유/강점/개선점/모범답안 모두 정제
        reasons = {k: _strip_foreign(str(v))
                   for k, v in (raw.get("reasons", {}) or {}).items()}
        strengths = [_strip_foreign(str(s)) for s in (raw.get("strengths", []) or [])]
        improvements = [_strip_foreign(str(i)) for i in (raw.get("improvements", []) or [])]
        model_answer = _strip_foreign(str(raw.get("model_answer", "") or ""))

        return EvaluationResult(
            scores=norm_scores,
            overall_score=round(overall, 1),
            grade=self._to_grade(overall),
            reasons=reasons,
            strengths=[s for s in strengths if s],
            improvements=[i for i in improvements if i],
            model_answer=model_answer,
            raw=raw,
        )

    @staticmethod
    def _as_int(v) -> int:
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return 0

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
    evaluator = ContentEvaluator()
    demo_question = "본인의 가장 큰 강점은 무엇이며, 그것을 발휘한 경험을 말해보세요."
    demo_answer = ("저는 책임감이 강합니다. 팀 프로젝트에서 맡은 일을 끝까지 해냈고, "
                   "팀원들이 저를 믿어주었습니다.")
    demo_role = "백엔드 개발자"
    try:
        print("평가 중입니다... (로컬 모델이라 처음엔 수십 초 걸릴 수 있어요)\n")
        result = evaluator.evaluate(demo_question, demo_answer, demo_role)
        result.pretty_print()
    except Exception as e:
        print(f"[에러] 평가 실패: {e}")
