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

@app.get("/chat")
def chat(request: Request, word : str):
    print('서버에서 받은 내용', word)
    result = Ollama_client(word)
    return templates.TemplateResponse("chat.html", context={"request": request, "result":result})

# if __name__ == '__main__':
#     uvicorn.run("app.main:app", host='127.0.0.1', port=8000, reload=True)