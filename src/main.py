from fastapi import FastAPI
from src.api.routes import upload, ask, health
from src.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Include routers
app.include_router(health.router, tags=["monitoring"])
app.include_router(upload.router, tags=["upload"])
app.include_router(ask.router, tags=["query"])

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API.",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
