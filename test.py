from fastapi import FastAPI
import asyncio
import uvicorn

app = FastAPI()
flag = False

async def simulate_long_running_task():
    global flag
    # Giả lập hoạt động mất thời gian
    await asyncio.sleep(5)

    flag = False
    return "Process completed!"

@app.get("/setup")
async def setup():
    global flag
    # Gọi hàm mất thời gian nhưng không chờ đợi nó hoàn thành
    return flag

@app.post("/process")
async def process():
    global flag
    flag = True
    await simulate_long_running_task()
    return {"is processing..."}

if __name__ == "__main__":
    uvicorn.run("test:app", host="127.0.0.1", port=8000, reload=True)