import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ollama_client import Ollama_client
import ollama
# 스트리밍 엔드포인트 (실시간 토큰 반환, 더 빠른 체감)
from fastapi.responses import StreamingResponse

# 응답을 JSON으로 해주는 부품을 만들자
# BaseModel의 모든 변수 함수를 확장해서 사용 상속
class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    message: str

# 사용할 모델 이름 (미리 ollama pull <model>로 다운로드 필요, 예: ollama pull llama3.2)
MODEL = "gemma3:1b"

app = FastAPI()

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates 설정
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


OLLAMA_BASE_URL = "http://localhost:11434"
@app.get("/health")
async def health_check():
    """FastAPI와 Ollama의 health 상태를 확인하는 엔드포인트"""
    try:
        # Ollama health check
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)

        if response.status_code == 200:
            ollama_status = "healthy"
            message = "fastapi & ollama 제대로 동작중"
        else:
            ollama_status = "unhealthy"
            message = f"Ollama returned status code: {response.status_code}"

    except httpx.ConnectError:
        ollama_status = "연결불가"
        message = "Ollama 연결할 수 없음."
    except httpx.TimeoutException:
        ollama_status = "타임아웃"
        message = "Ollama 타임 아웃"
    except Exception as e:
        ollama_status = "error"
        message = f"Error checking Ollama: {str(e)}"

    return HealthResponse(
        status='ok',
        ollama_status=ollama_status,
        message=message
    )

# 앱 시작 시 모델 미리 로드 (preload)
@app.on_event("startup")
async def preload_model():
    try:
        # 빈 프롬프트로 모델 로드 + 영구 유지
        await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=" ",  # 빈 프롬프트 (또는 "preload" 같은 더미 텍스트)
            keep_alive=-1  # -1: 영구적으로 메모리에 유지
        )
        print(f"{MODEL} 모델이 미리 로드되었습니다. (메모리에 영구 유지)")
    except Exception as e:
        print(f"모델 preload 실패: {e}")

# 일반 generate 엔드포인트 (스트리밍 없이 전체 응답)
@app.get("/chat")
async def generate(word : str, request : Request):
    try:
        response = await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=word,
            options={"temperature": 1},
            keep_alive=-1  # 필요 시 후속 요청에서도 유지
        )
        return templates.TemplateResponse("chat.html",
                                      context={"request": request,
                                               "result" : response["response"]
                                               })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ollama-test")
def ollama_test(request : Request):
    return templates.TemplateResponse("ollama-test.html",
                                      context={"request": request
                                               })

@app.get("/stream")
async def stream(word : str):
    return StreamingResponse(stream_generate(word), media_type="text/event-stream")

async def stream_generate(prompt: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        keep_alive=-1
    )
    async for part in stream:
        # "이 값을 내보내고, 여기서 잠깐 멈춰. 다음에 다시 불러주면 이어서 할게!"
        # ollama로 부터 받은 조각마다 보내..
        yield part["response"]


## 파라메타용 class 를 만들자.
##  파라메터 이름 똑같은 거 자동으로 변수에 들어감
## 다른 옵션값들 설정 가능
## BaseModel 이라는 클래스를 상속 받아서 만들어야 자동으로  이런 처리를 해줌
class SummarizeRequest(BaseModel):
    ## BaseModel(변수+함수) + 내가 추가한 변수
    text: str
    max_length : int = 200

@app.post("/summarize")
async def summarize(request : SummarizeRequest):
    # http://localhost:11434/api/generate, json=payload
    # post 방식으로 http요청을 해줌
    prompt = f"{request.text}를 {request.max_length}자로 요약해주세요"
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response) #dict
    return {'summary' : response['response'].strip()}


class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request : TranslateRequest):
    prompt = f"""
            {request.text}를 
            한국어로 번역해줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'translate': response['response'].strip()}

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
async def sentiment(request : SentimentRequest):
    prompt = f"""
            {request.text}의
            내용을 감정분석 해줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'sentiment': response['response'].strip()}


class BrainstormRequest(BaseModel):
        topic: str
        count: int = 3

@app.post("/brainstorm")
async def brainstorm(request : BrainstormRequest):
    prompt = f"""
            {request.topic} 주제에 대한 아이디어를
            {request.count} 개 만들어줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'brainstorm': response['response'].strip()}


class PoemRequest(BaseModel):
    topic: str
    style: str = '고조시'


@app.post("/poem")
async def poem(request: PoemRequest):
    prompt = f"""
            감성을 자극하는 시를 만들고 싶은데 {request.topic} 주제로
            {request.style}를(을) 작성해줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'poem': response['response'].strip()}


class RecipeRequest(BaseModel):
    ingredients: str
    servings: int = 2
    difficulty: str


@app.post("/recipe")
async def recipe(request: RecipeRequest):
    prompt = f"""
            {request.ingredients} 재료를 가지고
            난이도가 {request.difficulty},
            {request.servings} 인원 수 만큼의 요리를 만들 수 있는 레시피를 제공해줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'recipe': response['response'].strip()}



class CodingRequest(BaseModel):
    language: str
    differ: str = '보통으로'
    text: str
    count: int = 30


@app.post("/coding")
async def coding(request: CodingRequest):
    prompt = f"""
            {request.language} 프로그램 언어로
            난이도가 {request.differ} 작성된
            {request.text} 프로그램 코드를
            {request.count} 줄 이내로 작성해줘
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'coding': response['response'].strip()}

class NameRequest(BaseModel):
    # axios post 로 전달되는 키 이름이 같아야 한다.
    category: str = "카페"
    gender: str = "중성"
    count: int = 3
    vibe: str = "따뜻한"


@app.post("/names")
async def names(request : NameRequest):
    prompt = f"""
            {request.category}이름을 
            {request.gender}, {request.vibe}느낌으로 
            {request.count}개만 추천해줘
            결과화면은 다음과 같이 만들어줘
            1. 이름 - 간단 설명
            2. 이름 - 간단 설명
            3. 이름 - 간단 설명
            """
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------------")
    print(response)  # dict
    return {'names': response['response'].strip()}

# if __name__ == '__main__':
#     uvicorn.run("app.main:app", host='127.0.0.1', port=8000, reload=True)