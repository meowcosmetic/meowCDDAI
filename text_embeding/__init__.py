from .routes_upload import router as upload_router
from .routes_search import router as search_router
from .routes_admin import router as admin_router
from .routes_post import router as post_router
from .routes_description import router as description_router
from .routes_question import router as question_router
from .routes_csv import router as csv_router
from .routes_question_csv import router as question_csv_router

__all__ = [
    "upload_router",
    "search_router",
    "admin_router",
    "post_router",
    "description_router",
    "question_router",
    "csv_router",
    "question_csv_router",
]


