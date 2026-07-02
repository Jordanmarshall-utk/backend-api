from dotenv import load_dotenv

# Must run before any app module imports so os.getenv() module-level calls see .env values.
load_dotenv()

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api.v1 import api_router  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.graph.checkpointer import close_checkpointer, init_checkpointer  # noqa: E402
from app.graph.pipeline import init_pipeline  # noqa: E402
from app.services.memory_service import get_memory_manager  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_memory_manager()
    cp = await init_checkpointer()
    init_pipeline(cp)
    yield
    await close_checkpointer()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
