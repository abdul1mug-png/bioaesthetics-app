from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="BioAesthetic API",
        version="1.0.0",
        description="Biometric optimization platform.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        return response

    @app.on_event("startup")
    async def startup():
        from database import init_db
        import models  # ensure all models are registered
        await init_db()
        log.info("Database tables created.")

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": "1.0.0"}

    from routes import auth_router, skin_router, physique_router, hormone_router, dashboard_router
    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    app.include_router(skin_router, prefix="/skin", tags=["Skin"])
    app.include_router(physique_router, prefix="/physique", tags=["Physique"])
    app.include_router(hormone_router, prefix="/hormone", tags=["Hormone"])
    app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])

    return app


app = create_app()