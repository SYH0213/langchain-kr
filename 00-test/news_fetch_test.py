# filename: news_fetch_test.py
import sys
from urllib.parse import urlparse
import trafilatura

def fetch_news(url: str) -> dict:
    """
    주어진 뉴스 URL에서 본문 텍스트와 메타데이터를 추출해 반환.
    실패 시 빈 dict를 반환.
    """
    # 1) 원문 가져오기 (robots.txt 무시 옵션은 False 유지 권장)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {}

    # 2) 본문 추출 (메타데이터 포함)
    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_recall=True,     # 본문 누락 최소화
        with_metadata=True,    # 제목/작성일 등 메타데이터 포함
        output_format="json",  # JSON으로 받기(파싱 쉬움)
    )

    if not result:
        return {}

    # 3) trafilatura JSON 문자열 → dict
    import json
    data = json.loads(result)

    # 4) 유용 필드만 골라서 리턴
    return {
        "source": urlparse(url).netloc,
        "url": url,
        "title": data.get("title"),
        "date": data.get("date"),
        "author": data.get("author"),
        "text": data.get("text"),
        "language": data.get("language"),
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python news_fetch_test.py <뉴스_URL>")
        sys.exit(1)

    url = sys.argv[1]
    info = fetch_news(url)
    if not info:
        print("❌ 본문 추출 실패 (URL이 접근 제한이거나, 동적 렌더링이 강한 사이트일 수 있음)")
        sys.exit(1)

    print(f"제목: {info['title']}")
    print(f"날짜: {info['date']}")
    print(f"출처: {info['source']}")
    print("\n=== 본문 (앞 800자) ===")
    text_preview = (info["text"] or "").strip()
    print(text_preview[:800])
