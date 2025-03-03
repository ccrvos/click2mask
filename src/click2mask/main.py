from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from .models.inference import MaskGenerator
import debugpy

debugpy.listen(("0.0.0.0", 5679))

app = FastAPI(
    title="Click2Mask",
    description="Select object to segment using SAM-2",
    version="0.1.0",
)


# CSP headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' data:;"
    )
    return response


UPLOAD_DIR = Path("static/uploads")
MASKS_DIR = Path("static/masks")
UPLOAD_DIR.mkdir(exist_ok=True)
MASKS_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


mask_generator = MaskGenerator()


@app.get("/")
async def root(
    request: Request,
):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        session_id = str(uuid.uuid4())
        file_location = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(file.file.read())

        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "file_path": f"{file_location}",
                "session_id": session_id,
            },
        )

    except Exception as e:
        return {"error:": str(e)}


@app.post("/select-point")
async def select_point(
    request: Request,
    file_path: str = Form(...),
    click_x: int = Form(..., alias="click.x"),
    click_y: int = Form(..., alias="click.y"),
):

    try:
        return templates.TemplateResponse(
            "coordinate_display.html",
            {
                "request": request,
                "file_path": file_path,
                "x": click_x,
                "y": click_y,
            },
        )
    except Exception as e:
        return {"error:": str(e)}


@app.post("/generate-mask")
async def process_text(
    request: Request,
    file_path: str = Form(...),
    point_x: int = Form(...),
    point_y: int = Form(...),
):

    try:
        image = Image.open(Path(file_path))
        image_array = np.array(image.convert("RGB"))

        masked_image_png = mask_generator.process_point_mask(
            image_array, point_x, point_y
        )

        mask_filename = f"mask_{uuid.uuid4()}.png"
        mask_path = MASKS_DIR / mask_filename

        with open(mask_path, "wb") as f:
            f.write(masked_image_png)

        return templates.TemplateResponse(
            "mask_result.html",
            {
                "request": request,
                "file_path": file_path,
                "mask_path": f"/static/masks/{mask_filename}",
            },
        )

    except Exception as e:
        return {"error:": str(e)}
