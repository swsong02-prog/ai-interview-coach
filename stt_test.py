"""
stt_test.py
────────────────────────────────────────────────────────
음성 → 글자(STT) 단독 테스트  (Whisper 로컬)

이 파일이 하는 일 (딱 이것만):
  1. 마이크로 정해진 시간(기본 8초) 동안 녹음
  2. 녹음을 wav 파일로 저장
  3. Whisper 로 한국어 인식 → 글자로 변환해서 출력

여기서 STT가 잘 되는지 먼저 확인한 뒤, 모의면접 파일에 합칩니다.

사전 준비 (이미 설치했다면 통과)
  pip install openai-whisper sounddevice scipy
  winget install ffmpeg          (설치 후 새 터미널에서  ffmpeg -version  확인)

실행
  python stt_test.py
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import tempfile

# ── 의존성 import (없으면 친절히 안내) ──────────────────────
try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy 가 필요합니다:  pip install numpy") from e

try:
    import sounddevice as sd
except ImportError as e:
    raise ImportError(
        "sounddevice 가 필요합니다:  pip install sounddevice\n"
        "  (마이크 녹음을 담당합니다)"
    ) from e

try:
    from scipy.io.wavfile import write as wav_write
except ImportError as e:
    raise ImportError("scipy 가 필요합니다:  pip install scipy") from e

try:
    import whisper
except ImportError as e:
    raise ImportError(
        "openai-whisper 가 필요합니다:  pip install openai-whisper"
    ) from e


# ── 설정 ────────────────────────────────────────────────
RECORD_SECONDS = 8          # 녹음 시간(초). 길게 말할 거면 늘리세요.
SAMPLE_RATE = 16000         # Whisper 권장 샘플레이트(16kHz)
WHISPER_MODEL = "base"      # tiny < base < small < medium < large
                            #  - tiny/base: 빠르지만 정확도 낮음
                            #  - small: 한국어 균형 좋음(추천, 단 다운로드 좀 큼)
                            #  컴퓨터가 버거우면 "tiny", 정확도 원하면 "small"


def record_audio(seconds: int, sample_rate: int) -> np.ndarray:
    """마이크로 지정 시간 동안 녹음하여 numpy 배열로 반환."""
    print(f"\n● 녹음을 시작합니다. {seconds}초 동안 말해주세요...")
    print("  (예: '안녕하세요, 저는 백엔드 개발 직무에 지원한 홍길동입니다.')\n")

    try:
        # mono(1채널), float32 로 녹음
        audio = sd.rec(int(seconds * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype="float32")
        # 진행 상황을 1초 단위로 표시
        for remaining in range(seconds, 0, -1):
            print(f"  녹음 중... {remaining}초 남음", end="\r")
            sd.sleep(1000)   # 1초 대기(ms)
        sd.wait()            # 녹음 완료까지 대기
        print("  녹음 완료!                      ")
        return audio
    except Exception as e:
        raise RuntimeError(
            f"녹음에 실패했습니다: {e}\n"
            "  · 마이크가 연결되어 있는지 확인하세요.\n"
            "  · 윈도우 '설정 > 개인정보 > 마이크'에서 앱 접근이 허용됐는지 확인하세요."
        ) from e


def save_wav(audio: np.ndarray, sample_rate: int) -> str:
    """녹음 배열을 임시 wav 파일로 저장하고 경로 반환."""
    # float32(-1~1) → int16 로 변환(wav 표준)
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    tmp_path = os.path.join(tempfile.gettempdir(), "stt_test_record.wav")
    wav_write(tmp_path, sample_rate, audio_int16)
    return tmp_path


def transcribe(wav_path: str, model_name: str) -> str:
    """Whisper 로 wav 파일을 한국어 텍스트로 변환."""
    print(f"\n● Whisper 모델 로딩 중... ('{model_name}')  첫 실행이면 다운로드가 진행됩니다.")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Whisper 모델 로드 실패: {e}\n"
            "  · 인터넷 연결을 확인하세요(첫 실행 시 모델 다운로드 필요).\n"
            "  · 계속 실패하면 WHISPER_MODEL 을 'tiny' 로 바꿔보세요."
        ) from e

    print("● 음성을 글자로 변환 중...")
    try:
        # language='ko' 로 한국어 고정 → 정확도 향상
        result = model.transcribe(wav_path, language="ko", fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(
            f"음성 변환 실패: {e}\n"
            "  · ffmpeg 이 설치됐는지 확인하세요(터미널에서  ffmpeg -version )."
        ) from e


def main():
    print("=" * 52)
    print("  음성 → 글자(STT) 테스트  [Whisper 로컬]")
    print("=" * 52)

    # 1) 녹음
    audio = record_audio(RECORD_SECONDS, SAMPLE_RATE)

    # 녹음이 너무 조용하면(마이크 문제) 경고
    volume = float(np.abs(audio).mean())
    if volume < 0.001:
        print("\n[경고] 녹음된 소리가 거의 없습니다. 마이크 입력이 잡혔는지 확인하세요.")

    # 2) 저장
    wav_path = save_wav(audio, SAMPLE_RATE)
    print(f"  (임시 저장: {wav_path})")

    # 3) 변환
    text = transcribe(wav_path, WHISPER_MODEL)

    # 4) 결과
    print("\n" + "=" * 52)
    print("  인식 결과")
    print("=" * 52)
    if text:
        print(f"  \"{text}\"")
    else:
        print("  (인식된 내용이 없습니다. 더 크게/또렷이 말해보세요.)")
    print("=" * 52)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[안내] 사용자 중단(Ctrl+C).")
    except Exception as e:
        print(f"\n[에러] {e}")
        sys.exit(1)
