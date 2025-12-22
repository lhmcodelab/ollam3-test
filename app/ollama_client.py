# http 서버 11434 올라마 서버 연결 모듈
from fastapi import HTTPException
import requests

# hhttp 서버이므로 http 연결 모듈 필요
# 순서대로 호출해서 받을 것이면 requests, tablib.request
# 동시에 호출 받을 것이면 http
# Ollama 기본 설정
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"

def Ollama_client(word : str):
    if not word or word.strip() == "":
        raise HTTPException(status_code=400, detail="word값이 없다")
    # import httpx
    try :
        # 올라마 서버로 보낼 데이터 만들고
        # Ollama API에 전송할 페이로드
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": word.strip(),
            "stream": False,  # 스트리밍 비활성화로 전체 응답 받기
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        # 올라마 서버로 post 요청
        response = requests.post(url = f"{OLLAMA_BASE_URL}/api/generate", json=payload)

        print(response.json())
        # 결과 받아오면 결과를 전달함
        if response.status_code == 200:
            ollama_response = response.json()
            # JSON 응답 구성
            result = {
                "success": True,
                "question": word,
                "answer": ollama_response.get("response", ""),
                "model": ollama_response.get("model", DEFAULT_MODEL),
                "metadata": {
                    "total_duration": ollama_response.get("total_duration"),
                    "load_duration": ollama_response.get("load_duration"),
                    "prompt_eval_count": ollama_response.get("prompt_eval_count"),
                    "eval_count": ollama_response.get("eval_count"),
                    "eval_duration": ollama_response.get("eval_duration")
                }
            }
            return result
    except httpx.ConnectError:
        print("connect error")
    except httpx.HTTPError:
        print("통신 error")
    except httpx.TimeoutException:
        print("타임아웃 error")
    except Exception as e:
        print("올라마 서버와 통신 에러 발생함" + str(e))
        return {"success": False}