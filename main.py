print("🚀 MEOW CDD AI VERSION 2.0 STARTING (GPU ENABLED)...")
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from text_embeding import (
    upload_router, search_router, admin_router, post_router, description_router, 
    question_router, csv_router, question_csv_router,
    screening_gaze_router, screening_expression_router, screening_pose_router,
    screening_interaction_router, screening_speech_router,
    extraction_router, jobs_router
)


app = FastAPI(
    title="Book Vector Service",
    description="Service để lưu trữ và tìm kiếm vector của sách sử dụng Qdrant với Hybrid Search",
    version="2.0.0",
)

# CORS - use allowlist from env, reject wildcard
_cors_origins_raw = os.getenv("CORS_ORIGINS", "")
if _cors_origins_raw and _cors_origins_raw.strip() != "*":
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
else:
    _cors_origins = ["http://localhost:3101", "http://localhost", "http://127.0.0.1:3101"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)


# Mount feature routers
app.include_router(admin_router)
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(post_router)
app.include_router(description_router)
app.include_router(question_router)
app.include_router(csv_router)
app.include_router(question_csv_router)

# Mount screening routers
app.include_router(screening_gaze_router)
app.include_router(screening_expression_router)
app.include_router(screening_pose_router)
app.include_router(screening_interaction_router)
app.include_router(screening_speech_router)

# Mount extraction router
app.include_router(extraction_router)

# Mount job API router (async pipeline)
app.include_router(jobs_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
