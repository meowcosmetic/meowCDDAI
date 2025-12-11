from .routes_upload import router as upload_router
from .routes_search import router as search_router
from .routes_admin import router as admin_router
from .routes_post import router as post_router
from .routes_description import router as description_router
from .routes_question import router as question_router
from .routes_csv import router as csv_router
from .routes_question_csv import router as question_csv_router
from .routes_screening_gaze import router as screening_gaze_router
from .routes_screening_expression import router as screening_expression_router
from .routes_screening_pose import router as screening_pose_router
from .routes_screening_interaction import router as screening_interaction_router
from .routes_screening_speech import router as screening_speech_router

__all__ = [
    "upload_router",
    "search_router",
    "admin_router",
    "post_router",
    "description_router",
    "question_router",
    "csv_router",
    "question_csv_router",
    "screening_gaze_router",
    "screening_expression_router",
    "screening_pose_router",
    "screening_interaction_router",
    "screening_speech_router",
]


