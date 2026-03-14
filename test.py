from fastgradio import App

app = App()

@app.cpu(concurrency_limit=100)
def call_weather_api():
    return "sunny"


@app.gpu()  # or @app.gpu(rank=0)  # or spaces.GPU()
@app.mcp()
@app.api(name="get_image")
def run_text_to_image_model(weather: str) -> None:
    pass
    

@app.get("/")
async def root():
    weather = call_weather_api()
    image = run_text_to_image_model(weather)
    return {"weather": weather, "image": image}

app.launch()