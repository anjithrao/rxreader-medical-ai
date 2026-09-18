from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@router.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_manifest():
    return JSONResponse({})
