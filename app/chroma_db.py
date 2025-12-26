# chroma_db.py
from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

import requests
import chromadb
import fitz  # PyMuPDF

class ChromaRAG:
    ################# 1. 설정부분
    # ollama 설정
    # ollama url : http://localhost:11434
    # 인베딩 모델 : nomic-embed-text
    # 생성형 모델 : gemma3:1b
    def __init__(
            self,
            chroma_dir: str = "./chroma_data",
            collection_name: str = "rag_docs",
            ollama_base_url: str = "http://localhost:11434",
            embed_model: str = "nomic-embed-text",
            gen_model: str = "gemma3:1b",
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.gen_model = gen_model

    # chroma 설정
    # 폴더만든 것에 chroma db연결함 --> chroma_data
    # collection(table, 폴더)를 생성함
    #  ---> 명 : rag_docs
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection2 = self.client.get_or_create_collection(name=collection_name + str(2))

    def __str__(self):
        return f"client: {self.client}\nembed_model: {self.embed_model}\ngen_model: {self.gen_model}\ncollection: {self.collection}\ncollection2: {self.collection2})"

    ################# 2. 임베딩하고 ollama 요청해서 답변 받아오는 부분
    # embed
    def embed(self, text: str) -> List[float]: # 리턴타입!!
        # ollama 에 주소로 임베딩해 달라고 해아함
        url = f"{self.ollama_base_url}/api/embeddings"
        resp = requests.post(url, json={"model": self.embed_model, "prompt": text})
        print(resp.json()) # dict 형태로 만들어서 프린트
        data = resp.json()
        return data["embedding"]
    # 답생성 generate
    def generate(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.gen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                # 한글 2바이트 3바이트 30글자 정도
                "num_predict": 64
            }
        }
        r = requests.post(url, json=payload, timeout=120)
        print(r.json())
        data = r.json()
        return data['response']
    ########## 3. chuck 만드는 부분
    # 통으로 읽은 text를 작게 자르고(chuck, 조각)
    # -----------------------------
    # Chunking
    # -----------------------------
    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(0, end - overlap_chars)

        return chunks
    # pdf를 읽어서 text로 만들자

    # collection에 몇개 들어있는지 확인하는 함수
    # -----------------------------
    # Chroma operations
    # -----------------------------
    def count(self) -> int:
        return self.collection.count()

    def get_collection(self):
        results = self.collection.get(include=["documents", "metadatas"], limit=10)
        return results
    ######### 4. 크로마 db에 적재
    # 텍스트를 크로마 db에 적재하자(ingest)
    def ingest_texts(self, texts: List[str], source: str = 'manual'):
        if not texts:
            return 0
        for i, t in enumerate(texts):
            self.collection.add(
                ids = [str(uuid.uuid4())],
                documents=[t],
                embeddings=[self.embed(t)],
                metadatas=[{"chunk": i, "source": source}]
            )
        return len(texts)
    ####### 5. 질문하고 답변하는 부분
    # 질의하고 답변을 만드는 것과 관련된 함수를 생성

if __name__ == "__main__":
    rag = ChromaRAG()
    # print(rag)
    # rag.embed(text='hello')
    # print(rag.generate(prompt='한식'))

    text2 = '''
        이하늬 기획사 미등록 논란, 단순 실수일까? 60억 추징까지 이어진 이유
        최근 배우 이하늬를 둘러싼 논란이 잇따르며 대중의 이목이 집중되고 있다. 핵심은 연예 활동을 관리하던 법인이 대중문화예술기획업 미등록 상태로 운영됐다는 점이다. 단순 행정 착오로 보기엔 사안이 가볍지 않다는 평가도 나온다. 이번 사건을 차분히 정리해본다.
        이하늬 대중문화예술기획업 미등록, 어떤 혐의인가
        서울 강남경찰서는 이하늬와 남편, 그리고 법인 ‘호프프로젝트’를 대중문화예술산업발전법 위반 혐의로 검찰에 불구속 송치했다. 문제의 핵심은 2015년 설립된 법인이 연예 매니지먼트 성격의 활동을 하면서도 필수 절차인 기획업 등록을 하지 않았다는 점이다.
        이하늬 측은 당시 제도 인식이 부족했다고 해명했지만, 법률상 ‘몰랐다’는 사유는 면책 사유로 인정되기 어렵다. 이로 인해 이하늬 대중문화예술기획업 미등록 논란은 단순 해프닝을 넘어 법적 책임 문제로 확산됐다.
        법 위반 시 처벌 수위는 어느 정도일까
        현행법에 따르면 기획업 미등록 상태로 영업을 할 경우 2년 이하 징역 또는 2000만원 이하 벌금, 나아가 영업정지 처분까지 가능하다. 특히 연예인 개인 법인이라 하더라도 실제 매니지먼트 기능을 수행했다면 등록 의무에서 자유로울 수 없다.
        전문가들은 이하늬 대중문화예술기획업 미등록 사안이 장기간 지속됐다는 점에서 행정상 주의 수준을 넘어섰다고 보고 있다.
        고발인 “사회적 영향력 큰 만큼 책임 더 무겁다”
        이번 사안과 관련해 고발인은 “대중적 영향력이 큰 인물일수록 준법 책임은 더 엄격히 요구된다”고 지적했다. 개인 브랜드와 법인 영업이 결합된 구조라면, 법적 요건을 상시 점검할 의무가 있다는 주장이다.
        이 같은 문제 제기는 연예인 개인 법인 전반에 대한 경각심으로 이어지고 있으며, 이하늬 대중문화예술기획업 미등록 사례가 하나의 기준점이 될 수 있다는 평가도 나온다.
        60억 세금 추징까지… 논란이 커진 배경
        이번 사건은 과거 세무 이슈와 맞물리며 더욱 주목받고 있다. 이하늬는 지난해 법인 수익 처리 문제로 약 60억원의 세금 추징을 받은 바 있다. 상시근로자 없이 거액의 급여가 지급된 점, 소액 자본금으로 고가 부동산을 매입한 구조 등도 함께 도마에 올랐다.
        소속사는 “법 해석 차이”라고 해명했지만, 대중의 시선은 한층 엄격해진 상황이다. 이로 인해 이하늬 대중문화예술기획업 미등록 문제 역시 단독 이슈가 아닌, 종합적인 경영·준법 논란으로 인식되고 있다.
        향후 처분 가능성과 시사점
        법조계에서는 고의성이 낮고 뒤늦게나마 등록을 마쳤다면 벌금형이나 기소유예 가능성도 있다고 본다. 다만 이번 사건을 계기로 유사한 연예인 개인 법인들에 대한 점검이 강화될 가능성은 높다.
        결국 이하늬 대중문화예술기획업 미등록 사안은 한 배우 개인의 문제가 아니라, 연예 산업 전반의 구조와 법 인식 수준을 되돌아보게 하는 계기가 되고 있다.
    '''
    # result = ChromaRAG.chunk_text(text)
    # print(result)
    #
    # result2 = rag.embed(text = result[0])
    # print(result2)
    #
    # print(rag.generate(prompt = text))
    #
    # result3 = rag.ingest_texts(result)
    # print(result3)
    #
    # print(rag.count())
    # print(rag.get_collection())

    # text2 = '''
    # 위기를 헤쳐 나가기 위한 도전이 시작됐다고 미국의 유력일간지 뉴욕타임스(NYT)가 25일(현지시간) 보도했다.
    # NYT의 대중음악 담당 존 캐러매니카 기자는 이날 '내면의 악마와 싸운 K팝의 2025년'이라는 제목의 기사에서 '케이팝 데몬 헌터스' 등을 예로 들면서 K팝이 장르를 넘어 세계적 문화 코드로 자리 잡았다고 평가했다.
    # K팝이 지난 10여년간 경쟁자들과 비교되지 않을 정도로 강력한 혁신을 앞세워 대중음악의 판도를 바꿔왔다는 것이다.
    # 그러나 산업적 측면에서 K팝을 들여다보면 구조적 한계가 드러난다는 것이 캐러매니카 기자의 주장이다.
    # 소수 대형 기획사가 주도하는 고도로 체계화된 산업구조 속에서 독창성 있는 콘텐츠를 지속적으로 생산하는 것은 분명하게 한계가 있다는 이야기다.
    # 특히 올해는 '산업'으로서의 K팝과 '예술'로서의 K팝 사이의 균열이 부각된 한해라는 설명이다.
    # 캐러매니카 기자는 '최근 수년간 가장 혁신적인 그룹'이라는 평가와 함께 뉴진스와 소속 기획사 어도어의 법적 분쟁을 사례로 들었다.
    # 그는 "이런 상황에서 뉴진스의 재출발은 민희진 어도어 전 대표와의 결별이나 경직된 환경 탓에 이전만큼 혁신적이거나 만족스럽지 않을 가능성이 크다"고 내다봤다.
    # 법원이 어도어의 손을 들어준 이후 뉴진스 멤버들의 복귀가 발표됐지만 여전히 완전체 활동에 대한 소식이 들리지 않고 있다는 것이다.
    # 현재 K팝은 창의적 측면에선 막다른 골목에 도달했다는 것이 NYT의 분석이다.
    # 스트레이 키즈와 트와이스, 엔하이픈, 세븐틴은 상업적으로 성공했지만, 음악적 틀이 점점 진부해지고 있다는 것이다.
    # 스파이스 걸스와 카일리 미노그 등 영어권 팝스타들이 자신의 대표곡을 K팝 스타일로 재해석하고 K팝 아이돌과 협업하는 애플TV플러스(+)의 '케이팝드' 같은 프로그램도 장르가 포화 상태에 이르렀을 때 나타나는 현상이라는 설명이다.
    # 다만 NYT는 기존 K팝 산업계 바깥의 한국 대중음악에 주목했다.
    # 에피와 더딥 등 하이퍼팝 계열의 뮤지션과 프로듀서 kimj를 언급하면서 "대기업 시스템 밖의 한국 음악계에선 활발한 혁신이 이뤄지고 있다. 이들은 현재 가장 도발적인 음악을 만들어 내고 있다"고 평가했다.
    # 캐러매니카 기자는 "K팝 산업이 내부의 피로와 불안과 싸우고 있는 동안, K팝 체제를 전복할 사운드는 이미 태어났을지도 모른다"며 글을 맺었다.
    # '''
    # 테스트를 청크한다.
    chunk_text2 = ChromaRAG.chunk_text(text2)
    print(f"chaunk text : {chunk_text2}")

    len = rag.ingest_texts(chunk_text2)
    print(f"embed len : {len}")
    print(f"embed total len : {rag.count()}")
    print(rag.get_collection())