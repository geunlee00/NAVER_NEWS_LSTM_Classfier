"""BBC 기사 분류 실습용 샘플 데이터 생성 모듈."""

from __future__ import annotations

import os
import time
from typing import List, Tuple


# BBC 기사 분류를 목표로 하며 sport, business, politics, tech, entertainment 계열의 라벨을 다룬다.
# 실제 BBC 원본 데이터 파일이 없는 환경에서도 실행되도록 각 분야별 영문 샘플 기사를 내장했다.
SAMPLE_DATA: List[Tuple[str, str]] = [
    ("The football team won the final after scoring two late goals in the stadium", "sport"),
    ("The tennis champion reached the semi final with a powerful serve and fast return", "sport"),
    ("A young striker signed a new contract before the league match this weekend", "sport"),
    ("The coach praised the players after the club moved to the top of the table", "sport"),
    ("Olympic athletes trained hard for the swimming race and cycling event", "sport"),
    ("The central bank increased interest rates to slow inflation and protect the economy", "business"),
    ("Shares rose sharply after the company reported strong quarterly profit", "business"),
    ("The airline announced job cuts as fuel prices increased across the market", "business"),
    ("Investors watched the stock market closely after the trade deal was announced", "business"),
    ("Retail sales improved as consumers spent more during the holiday season", "business"),
    ("The prime minister answered questions about the new education policy in parliament", "politics"),
    ("Voters will choose a new president after weeks of national election debate", "politics"),
    ("The government proposed a law to reform public health services", "politics"),
    ("Opposition leaders criticised the budget plan during a televised speech", "politics"),
    ("The mayor promised to improve transport and housing after winning the vote", "politics"),
    ("The smartphone maker released a new device with faster artificial intelligence features", "tech"),
    ("Researchers developed software that detects security attacks on cloud servers", "tech"),
    ("A startup built a robot that can learn from voice commands and camera images", "tech"),
    ("The company updated its mobile app with improved privacy controls", "tech"),
    ("Scientists used machine learning to analyse large amounts of online text", "tech"),
    ("The actor won an award for a drama film at the international festival", "entertainment"),
    ("The singer released a new album after a successful concert tour", "entertainment"),
    ("A popular television show returned with new characters and a surprising story", "entertainment"),
    ("The movie director announced the release date for a comedy sequel", "entertainment"),
    ("Fans watched the music performance live during the weekend broadcast", "entertainment"),
]

#
# def load_sample_data() -> Tuple[List[str], List[str]]:
#     """내장 샘플 데이터를 기사 문장 목록과 라벨 목록으로 분리해서 반환한다."""
#
#     texts = [text for text, _ in SAMPLE_DATA]    # 각 튜플의 첫 번째 값인 기사 문장만 모은다.
#     labels = [label for _, label in SAMPLE_DATA] # 각 튜플의 두 번째 값인 카테고리 라벨만 모은다.
#     return texts, labels                         # 모델 학습에 사용할 입력 데이터와 정답 데이터를 반환한다.


def load_sample_data(
    max_per_category: int = 40,
) -> Tuple[List[str], List[str]]:
    """저장된 데이터 파일이 있으면 불러오고, 없으면 네이버 뉴스를 크롤링하여 저장 후 반환한다.

    Args:
        max_per_category: 카테고리별로 수집할 최대 기사 수 (크롤링 시에만 사용).

    Returns:
        texts:  각 기사의 본문 문자열 목록.
        labels: texts 와 동일한 인덱스에 대응하는 카테고리 라벨 목록.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    texts_path = os.path.join(data_dir, "texts.txt")
    labels_path = os.path.join(data_dir, "labels.txt")

    # 저장된 파일이 있으면 크롤링 없이 바로 반환
    if os.path.exists(texts_path) and os.path.exists(labels_path):
        with open(texts_path, encoding="utf-8") as f:
            texts = f.read().splitlines()
        with open(labels_path, encoding="utf-8") as f:
            labels = f.read().splitlines()
        print(f"저장된 데이터 로드 완료 (총 {len(texts)}건)")
        return texts, labels

    # 파일이 없으면 크롤링 시작
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError("pip install requests beautifulsoup4 를 먼저 실행하세요.") from e

    categories: dict[str, int] = {
        "정치":   100,
        "경제":   101,
        "사회":   102,
        "생활문화": 103,
        "세계":   104,
        "IT과학":  105,
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    texts: List[str] = []
    labels: List[str] = []

    for label, sid1 in categories.items():
        list_url = f"https://news.naver.com/section/{sid1}"

        try:
            resp = requests.get(list_url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[{label}] 목록 페이지 요청 실패: {e}")
            continue

        collected = 0
        for tag in soup.select(".sa_text_strong"):
            if collected >= max_per_category:
                break
            text = tag.get_text(strip=True)
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            collected += 1

        print(f"[{label}] {collected}건 수집 완료")
        time.sleep(0.5)  # 카테고리 요청 사이 대기

    # 수집 결과를 파일로 저장
    os.makedirs(data_dir, exist_ok=True)

    with open(texts_path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    print(f"저장 완료: {data_dir} (총 {len(texts)}건)")

    return texts, labels
