"""Web pages, API routes, and application startup."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from exam_src.api.deps import DEV_USER
from exam_src.api.routers import exams, grading, results, utility
from exam_src.db.session import init_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "exam_web/templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    ## Resume saved jobs after a restart.
    from exam_src.api.routers.grading import resume_incomplete_jobs

    resume_incomplete_jobs()

    ## load the model once at startup, never per-request.
    try:
        from exam_cnn.models.inference import LetterInferenceWrapper

        app.state.inference_wrapper = LetterInferenceWrapper()
    except FileNotFoundError:
        ## OMR still works when the optional letter model is missing.
        app.state.inference_wrapper = None

    yield


app = FastAPI(title="AEGS - Automatic Exam Grading System", lifespan=lifespan)


@app.middleware("http")
async def no_cache_headers(request: Request, call_next):
    """Show the latest pages and scripts after a reload."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "exam_web/static")),
    name="static",
)

app.include_router(exams.router)
app.include_router(grading.router)
app.include_router(results.router)
app.include_router(utility.router)


def render_page(request: Request, template: str, **context):
    page_data = {"user": DEV_USER}
    page_data.update(context)
    return templates.TemplateResponse(request, template, page_data)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return RedirectResponse("/dashboard")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return render_page(request, "dashboard.html")


@app.get("/exams/new", response_class=HTMLResponse)
def exam_create_page(request: Request):
    return render_page(request, "exam_create.html", answer_key_mode=False)


@app.get("/answer-keys/new", response_class=HTMLResponse)
def answer_key_create_page(request: Request):
    return render_page(request, "exam_create.html", answer_key_mode=True)


@app.get("/exams/{exam_id}/answer-key", response_class=HTMLResponse)
def answer_key_edit_page(request: Request, exam_id: int):
    """Enter, replace or copy an answer key for an existing graded exam."""
    return render_page(
        request, "exam_create.html", answer_key_mode=True, edit_exam_id=exam_id
    )


@app.get("/exams/{exam_id}/upload", response_class=HTMLResponse)
def upload_page(request: Request, exam_id: int):
    return render_page(request, "upload.html", exam_id=exam_id)


@app.get("/exams/{exam_id}/review", response_class=HTMLResponse)
def review_page(request: Request, exam_id: int):
    return render_page(request, "review.html", exam_id=exam_id)


@app.get("/exams/{exam_id}/results", response_class=HTMLResponse)
def results_page(request: Request, exam_id: int):
    return render_page(request, "results.html", exam_id=exam_id)
