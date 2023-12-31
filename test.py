from fastapi import FastAPI, BackgroundTasks
import random
import uvicorn

app = FastAPI()

flag = False

async def simulate_long_running_task(a):
    global flag

    for i in range(10**8):
        a += random.random()

    flag = False
    print("completed")
    return None

@app.post("/setup")
async def setup():
    global flag
    return flag 

@app.post("/process")
async def process(background_tasks: BackgroundTasks):
    global flag
    flag = True

    test()
    background_tasks.add_task(test)

    return {"message": "Process started in the background."}

if __name__ == "__main__":
    uvicorn.run("test:app", host="127.0.0.1", port=8000, reload=True)

async def test():
    await simulate_long_running_task(a=0)
    return None