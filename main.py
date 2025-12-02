from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from text_embeding import upload_router, search_router, admin_router, post_router, description_router, question_router, csv_router, question_csv_router


app = FastAPI(
    title="Book Vector Service",
    description="Service để lưu trữ và tìm kiếm vector của sách sử dụng Qdrant với Hybrid Search",
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
