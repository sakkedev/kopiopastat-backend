from fastapi import FastAPI, HTTPException, Query, Request, Depends, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import json
import time
import threading
import random
import unicodedata
import re
import secrets
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from locales import translate
import os
import shutil
from dotenv import load_dotenv
from datetime import datetime, date
import requests
from PIL import Image
import io

load_dotenv()
TOKENS = os.getenv("TOKENS").split(',') if os.getenv("TOKENS") else []
CAPTCHA_QUESTIONS = json.loads(os.getenv("CAPTCHA_QUESTIONS", "[]"))

# Create necessary directories at startup
os.makedirs("data", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

FILENAME: str = 'data/kopiopasta.json'

user_tokens = {}  # token: {"ip": str, "last_used": float}
captcha_tokens = {}  # token: {"ip": str, "expiration": float}
token_lock = threading.Lock()

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/actions_{start_time}.log"

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator

def clean_expired_tokens():
    now = time.time()
    with token_lock:
        to_delete = [t for t, data in user_tokens.items() if now - data["last_used"] > 7 * 24 * 3600]
        for t in to_delete:
            del user_tokens[t]
        to_delete_captcha = [t for t, data in captcha_tokens.items() if now > data["expiration"]]
        for t in to_delete_captcha:
            del captcha_tokens[t]

def get_current_token(request: Request):
    clean_expired_tokens()
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[7:]
    with token_lock:
        if token not in user_tokens:
            raise HTTPException(status_code=401, detail="Invalid token")
        if user_tokens[token]["ip"] != request.client.host:
            raise HTTPException(status_code=401, detail="Token not for this user")
        user_tokens[token]["last_used"] = time.time()
    return token

def get_captcha_token(request: Request):
    return request.headers.get("X-Captcha")

def log_action(ip: str, action: str, content: str):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {ip} | {action} | {content}\n"
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(log_entry)

def create_daily_backup():
    today = date.today().strftime("%Y-%m-%d")
    backup_filename = f"backups/kopiopasta_{today}.json"
    os.makedirs("backups", exist_ok=True)
    shutil.copy(FILENAME, backup_filename)

def daily_backup_loop():
    while True:
        time.sleep(24 * 3600)  # 24 hours
        create_daily_backup()

def verify_captcha(request: Request) -> bool:
    token: str = request.headers.get("X-Captcha")
    clean_expired_tokens()
    with token_lock:
        if token in captcha_tokens:
            if captcha_tokens[token]["ip"] == request.client.host and time.time() < captcha_tokens[token]["expiration"]:
                return True
    return False

def process_image(entry_id: int, filename: str, file: UploadFile, check_existing: bool = True):
    entry = pastaloader.get(entry_id)
    if check_existing and "image" in entry:
        raise HTTPException(status_code=400, detail="Entry already has an image")
    file_content = file.file.read()
    file_size = len(file_content)
    if file_size > 5 * 1024 * 1024:  # 5 MB max for processing
        raise HTTPException(status_code=400, detail="Image too large")
    file.file.seek(0)
    image = Image.open(file.file)
    if image.format not in ['JPEG', 'PNG', 'AVIF']:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    converted = False
    if image.format == 'AVIF':
        image = image.convert('RGB')
        save_format = 'JPEG'
        ext = '.jpg'
        converted = True
    elif image.format == 'PNG':
        if file_size < 500 * 1024:  # <500 KB PNG, keep as PNG
            save_format = 'PNG'
            ext = '.png'
        elif file_size > 1 * 1024 * 1024:  # >1MB PNG, try convert to JPG
            # Try converting to JPG
            jpg_image = image.convert('RGB')
            jpg_buffer = io.BytesIO()
            jpg_image.save(jpg_buffer, format='JPEG')
            jpg_size = len(jpg_buffer.getvalue())
            if jpg_size < 1 * 1024 * 1024:  # JPG <1MB, use it
                image = jpg_image
                save_format = 'JPEG'
                ext = '.jpg'
                converted = True
            else:
                raise HTTPException(status_code=400, detail="Image too large even after conversion")
        else:  # 500KB to 1MB PNG, keep as PNG
            save_format = 'PNG'
            ext = '.png'
    else:  # JPEG
        save_format = 'JPEG'
        ext = '.jpg'
    
    # Check final size
    if save_format == 'JPEG':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        final_size = len(buffer.getvalue())
    else:
        final_size = file_size
    if final_size > 1 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")
    
    name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    saved_filename = f"{entry_id}_{name_without_ext}{ext}"
    if save_format == 'JPEG':
        image.save(os.path.join("data/images", saved_filename), 'JPEG')
    else:
        with open(os.path.join("data/images", saved_filename), 'wb') as f:
            f.write(file_content)
    
    # Set filename in JSON: include new extension if converted
    json_filename = name_without_ext + ext if converted else filename
    timestamp = int(time.time())
    entry["image"] = {"filename": json_filename, "timestamp": timestamp}
    pastaloader.save_data()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://kopiopastat.org"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type","Set-Cookie", "Authorization", "X-Captcha"],
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class PastaLoader:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.data_lock: threading.Lock = threading.Lock()
        self.normalized_data: list[dict[str, str | int]] = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.data: list[dict[str, str | int | list[dict[str, str | int]]]] = json.load(f)
        for item in self.data:
            item["contents"].sort(key=lambda x: x["timestamp"])
            if "found_in_google" not in item:
                item["found_in_google"] = True
        self.normalize_data()
        self.alphabetical_order: list[dict[str, str | int | list[dict[str, str | int]]]] = []
        self.organize_data()
        self.version: int = 1

    @timeout(10)
    def normalize_data(self):
        self.normalized_data = []
        for item in self.data:
            self.normalized_data.append({
                'content': self.normalize_text(item["contents"][-1]["content"]),
                'title': self.normalize_text(item["title"]),
                'id': item["id"]
            })

    @timeout(10)
    def save_data(self):
        with self.data_lock:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            self.normalize_data()
            self.organize_data()

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    @staticmethod
    def finnish_sort_key(text: str) -> tuple[int, ...]:
        text = text.lower()
        collation = 'abcdefghijklmnopqrstuvwxyzåäö'
        return tuple(collation.index(c) if c in collation else len(collation) + ord(c) for c in text)

    @timeout(10)
    def organize_data(self) -> None:
        self.alphabetical_order = sorted(self.data, key=lambda x: self.finnish_sort_key(x["title"]))

    def get(self, id: int) -> dict[str, str | int | list[dict[str, str | int]]]:
        """Returns the entry with matching id from data.

        Raises:
            ValueError if not found.
        """
        with self.data_lock:
            for item in self.data:
                if item["id"] == id:
                    return item
        raise ValueError(translate("entry_not_found"))

    def random(self) -> dict[str, str | int | list[dict[str, str | int]]]:
        ids = [item["id"] for item in self.data]
        random_id = random.choice(ids)
        return self.get(random_id)

    @timeout(10)
    def browse(self, start: int, end: int) -> list[dict[str, str | int]]:
        """Returns maximum 100 of alphabetical order between start and zero.
        Returns the latest content of each.
        """
        if end - start > 100 or start > end:
            raise ValueError(translate("invalid_range"))
        try:
            with self.data_lock:
                real_data = [item for item in self.alphabetical_order if item["found_in_google"]]
                items = real_data
        except IndexError:
            return []
        result = []
        for item in items:
            latest_content = item["contents"][-1]
            result.append({
                "title": item["title"],
                "id": item["id"],
                "content": latest_content["content"]
            })
        return result

    @timeout(10)
    def edit_entry(self, id, content: str, title: str, found_in_google: bool) -> int:
        content = content.rstrip('\n').lstrip('\n')
        if not content or len(content) < 5 or len(content) > 250000:
            raise ValueError(translate("content_must_be_5_250000_chars"))
        timestamp: int = int(time.time())
        entry: dict[str, str | int | list[dict[str, str | int]]] = self.get(id)
        title: str = title or entry["title"]
        if entry["title"] != title:
            with self.data_lock:
                entry["title"] = title
            self.version += 1
        if entry["found_in_google"] != found_in_google:
            with self.data_lock:
                entry["found_in_google"] = found_in_google
            self.version += 1
        if entry["contents"][-1]["content"] == content:
            return id

        with self.data_lock:
            entry["contents"].append({
                "timestamp": timestamp,
                "content": content
            })
        self.save_data()
        self.version += 1
        return id

    @timeout(10)
    def new_entry(self, title: str, content: str, found_in_google: bool = False) -> int:
        content = content.rstrip('\n').lstrip('\n')
        content = content.replace("\r\n", "\n")
        if not title or len(title) > 128:
            raise ValueError(translate("title_must_be_1_128_chars"))
        if not content or len(content) < 5 or len(content) > 250000:
            raise ValueError(translate("content_must_be_5_250000_chars"))

        normalized_title = self.normalize_text(title)
        with self.data_lock:
            if len([x for x in self.normalized_data if normalized_title == x["title"]]):
                raise ValueError(translate("title_already_exists"))
            new_id: int = max((item["id"] for item in self.data), default=0) + 1
            self.data.append({
                "title": title,
                "id": new_id,
                "contents": [{
                    "content": content,
                    "timestamp": int(time.time())
                }],
                "found_in_google": found_in_google
            })
        self.save_data()
        if not found_in_google:
            self.version -= 1
        return new_id

    @timeout(10)
    def search(self, q: str) -> list[dict[str, str | int]]:
        normalized_query = self.normalize_text(q)
        words: list[str] = normalized_query.split()
        matches: list[tuple[int, dict[str, str | int]]] = []
        for item in self.normalized_data:
            score = 5 if item["title"].startswith(normalized_query) else \
                4 if item["content"].startswith(normalized_query) else \
                3 if normalized_query in item["title"] else \
                2 if normalized_query in item["content"] else\
                0
            if score == 0:
                score = 1
                for word in words:
                    if word not in item["content"].split():
                        score = 0
                        break
            if score > 0:
                entry = self.get(item["id"])
                matches.append((score, {"title": entry["title"], "id": item["id"], "content": entry["contents"][-1]["content"]}))
        matches.sort(key=lambda x: x[0], reverse=True)
        result = [match[1] for match in matches[:10]]
        return result

    @timeout(10)
    def get_recent_edits(self, start: int, end: int) -> list[dict[str, str | int]]:
        if end - start > 100 or start > end:
            raise ValueError(translate("invalid_range"))
        all_events = []
        with self.data_lock:
            for item in self.data:
                for i, content in enumerate(item["contents"]):
                    event_type = "creation" if i == 0 else "edit"
                    all_events.append({
                        "id": item["id"],
                        "timestamp": content["timestamp"],
                        "title": item["title"],
                        "content": content["content"],
                        "type": event_type,
                        "found_in_google": item.get("found_in_google", True)
                    })
                if "image" in item:
                    all_events.append({
                        "id": item["id"],
                        "timestamp": item["image"]["timestamp"],
                        "title": item["title"],
                        "content": item["image"]["filename"],
                        "type": "image_added",
                        "found_in_google": item.get("found_in_google", True)
                    })
        all_events.sort(key=lambda x: x["timestamp"], reverse=True)
        try:
            return all_events[start:end]
        except IndexError:
            return []

    @timeout(10)
    def delete_content(self, id: int, timestamp: int) -> tuple[str, str]:
        entry = self.get(id)
        deleted_content = None
        with self.data_lock:
            original_length = len(entry["contents"])
            for c in entry["contents"]:
                if c["timestamp"] == timestamp:
                    deleted_content = c["content"]
                    break
            entry["contents"] = [c for c in entry["contents"] if c["timestamp"] != timestamp]
            if len(entry["contents"]) == original_length:
                raise ValueError(translate("content_not_found"))
            if not entry["contents"]:
                if "image" in entry:
                    image_filename = entry["image"]["filename"]
                    image_path = os.path.join("data/images", f"{entry['id']}_{image_filename}")
                    if os.path.exists(image_path):
                        os.remove(image_path)
                self.data.remove(entry)

        self.save_data()
        self.version += 1
        if not entry["contents"]:
            return "article", deleted_content
        else:
            return "content", deleted_content

    @timeout(10)
    def delete_image(self, id: int) -> None:
        entry = self.get(id)
        if "image" not in entry:
            raise ValueError("No image to delete")
        image_filename = entry["image"]["filename"]
        image_path = os.path.join("data/images", f"{entry['id']}_{image_filename}")
        if os.path.exists(image_path):
            os.remove(image_path)
        with self.data_lock:
            del entry["image"]
        self.save_data()
        self.version += 1




pastaloader = PastaLoader(FILENAME)

# Check at startup if today's backup exists, if not, create it
today = date.today().strftime("%Y-%m-%d")
backup_filename = f"backups/kopiopasta_{today}.json"
if not os.path.exists(backup_filename):
    create_daily_backup()

# Start background thread for daily backups
threading.Thread(target=daily_backup_loop, daemon=True).start()

# Models
class ContentItem(BaseModel):
    title: str
    id: int
    content: str

class BrowseResponse(BaseModel):
    version: int
    contents: list[ContentItem]

class GetResponse(BaseModel):
    title: str
    id: int
    content: str
    timestamp: int
    num_contents: int
    order_index: int
    last_in_order: int
    filename: str | None = None
    found_in_google: bool

class IdResponse(BaseModel):
    id: int

class HistoryResponse(BaseModel):
    title: str
    id: int
    contents: list[dict[str, str | int]]

class EditRequest(BaseModel):
    id: int
    content: str
    title: str
    found_in_google: bool = False

class NewRequest(BaseModel):
    title: str
    content: str

class SuccessResponse(BaseModel):
    message: str
    id: int

class RecentEditResponse(BaseModel):
    id: int
    timestamp: int
    title: str
    content: str
    type: str
    found_in_google: bool

class LoginRequest(BaseModel):
    code: str

class DeleteRequest(BaseModel):
    id: int
    timestamp: int

class DeleteImageRequest(BaseModel):
    id: int

class DataVersionResponse(BaseModel):
    version: int

class SearchResponse(BaseModel):
    title: str
    id: int
    content: str

class CaptchaQuestionResponse(BaseModel):
    question: str
    index: int

class CaptchaAnswerRequest(BaseModel):
    answer: str
    index: int

class CaptchaAnswerResponse(BaseModel):
    token: str

# Endpoints
@app.get("/browse", response_model=BrowseResponse)
@limiter.limit("30/minute")
def browse(request: Request, start: int = Query(..., ge=0), end: int = Query(..., ge=0)):
    try:
        contents = pastaloader.browse(start, end)
        return {"version": pastaloader.version, "contents": contents}
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/pasta", response_model=GetResponse)
@limiter.limit("180/minute")
def pasta(request: Request, id: int):
    try:
        entry = pastaloader.get(id)

        order_index = next((i for i, item in enumerate(pastaloader.alphabetical_order) if item["id"] == id), -1)
        filename = entry.get("image", {}).get("filename") if "image" in entry else None
        return {
            "title": entry["title"],
            "id": entry["id"],
            "content": entry["contents"][-1]["content"],
            "timestamp": entry["contents"][-1]["timestamp"],
            "num_contents": len(entry["contents"]),
            "order_index": order_index,
            "last_in_order": int(order_index == len(pastaloader.alphabetical_order) - 1),
            "filename": filename,
            "found_in_google": entry.get("found_in_google", True)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/get_by_order", response_model=IdResponse)
@limiter.limit("100/minute")
def get_by_order(request: Request, order: int = Query(..., ge=0)):
    if order >= len(pastaloader.alphabetical_order):
        raise HTTPException(status_code=400, detail=translate("order_index_out_of_range"))
    entry = pastaloader.alphabetical_order[order]
    return {"id": entry["id"]}

@app.get("/random", response_model=GetResponse)
@limiter.limit("120/minute")
def get_random(request: Request):
    entry = pastaloader.random()
    order_index = next((i for i, item in enumerate(pastaloader.alphabetical_order) if item["id"] == entry["id"]), -1)
    filename = entry.get("image", {}).get("filename") if "image" in entry else None
    return {
        "title": entry["title"],
        "id": entry["id"],
        "content": entry["contents"][-1]["content"],
        "timestamp": entry["contents"][-1]["timestamp"],
        "num_contents": len(entry["contents"]),
        "order_index": order_index,
        "last_in_order": int(order_index == len(pastaloader.alphabetical_order) - 1),
        "filename": filename,
        "found_in_google": entry.get("found_in_google", True)
    }

@app.get("/history", response_model=HistoryResponse)
@limiter.limit("50/minute")
def history(request: Request, id: int):
    try:
        return pastaloader.get(id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/download_backup")
@limiter.limit("5/hour")
def download_backup(request: Request):
    with pastaloader.data_lock:
        return FileResponse(
            path=FILENAME,
            media_type='application/json',
            filename='kopiopasta_backup.json',
            headers={"Content-Disposition": "attachment; filename=kopiopasta_backup.json"}
        )

@app.get("/images/{id}/{filename}")
@limiter.limit("1000/minute")
def get_image(request: Request, id: int, filename: str):
    path = f"data/images/{id}_{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    response = FileResponse(path=path, media_type='image/jpeg' if filename.endswith('.jpg') else 'image/png')
    response.headers["Access-Control-Allow-Origin"] = "https://kopiopastat.org"
    response.headers["Access-Control-Credentials"] = "true"
    return response

@app.post("/edit", response_model=SuccessResponse)
@limiter.limit("30/hour")
def edit(request: Request, req: EditRequest, token: str = Depends(get_current_token)):
    try:
        entry = pastaloader.get(req.id)
        pastaloader.edit_entry(req.id, req.content, req.title, req.found_in_google)
        updated_entry = pastaloader.get(req.id)
        edit_timestamp = updated_entry["contents"][-1]["timestamp"]
        log_action(request.client.host, "edit", f"Edited '{entry['title']}' (id: {req.id}) at {edit_timestamp} to: {req.content}")
        return {"message": translate("edit_successful"), "id": req.id}
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/new", response_model=SuccessResponse)
@limiter.limit("25/hour")
def new_entry(request: Request, title: str = Form(...), content: str = Form(...), file: UploadFile = File(None), found_in_google: bool = Form(False)):
    if not verify_captcha(request):
        try:
            get_current_token(request)
        except HTTPException:
            raise HTTPException(status_code=400, detail="Invalid CAPTCHA")
    try:
        id: int = pastaloader.new_entry(title, content, found_in_google)
        entry = pastaloader.get(id)
        create_timestamp = entry["contents"][0]["timestamp"]
        log_action(request.client.host, "new", f"Created new entry '{title}' (id: {id}) at {create_timestamp} with content: {content}")
        if file:
            process_image(id, file.filename, file, check_existing=False)
            log_action(request.client.host, "upload_image", f"Uploaded image '{file.filename}' for entry '{title}' (id: {id})")
        return {"message": translate("new_entry_created"), "id": id}
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/search", response_model=list[SearchResponse])
@limiter.limit("300/minute")
def search(request: Request, q: str = Query(..., min_length=3)):
    try:
        return pastaloader.search(q)
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=translate("request_timed_out"))

@app.get("/recent_edits", response_model=list[RecentEditResponse])
@limiter.limit("100/minute")
def recent_edits(request: Request, start: int = Query(..., ge=0), end: int = Query(..., ge=0)):
    try:
        return pastaloader.get_recent_edits(start, end)
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
@limiter.limit("50/hour")
def login(request: Request, req: LoginRequest):
    if req.code in TOKENS:
        token = secrets.token_urlsafe(32)
        ip = request.client.host
        with token_lock:
            user_tokens[token] = {"ip": ip, "last_used": time.time()}
        response = JSONResponse(content={"message": "Login successful", "token": token})
        return response
    else:
        raise HTTPException(status_code=401, detail="Invalid code")

@app.post("/logout")
@limiter.limit("50/hour")
def logout(request: Request, token: str = Depends(get_current_token)):
    with token_lock:
        del user_tokens[token]
    return {"message": "Logged out"}

@app.post("/delete")
@limiter.limit("100/hour")
def delete(request: Request, req: DeleteRequest, token: str = Depends(get_current_token)):
    try:
        entry = pastaloader.get(req.id)
        deleted_type, deleted_content = pastaloader.delete_content(req.id, req.timestamp)
        log_action(request.client.host, "delete", f"Deleted {deleted_type} from '{entry['title']}' (id: {req.id}): {deleted_content}")
        if deleted_type == "content":
            message = "Content deleted"
        else:
            message = "Article deleted"
        return {"message": message, "type": message}
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload_image", response_model=SuccessResponse)
@limiter.limit("20/hour")
def upload_image(request: Request, id: int = Form(...), filename: str = Form(...), file: UploadFile = File(...)):
    if not verify_captcha(request):
        try:
            get_current_token(request)
        except HTTPException:
            raise HTTPException(status_code=400, detail="Invalid CAPTCHA")
    try:
        process_image(id, filename, file, check_existing=True)
        entry = pastaloader.get(id)
        log_action(request.client.host, "upload_image", f"Uploaded image '{filename}' for entry '{entry['title']}' (id: {id})")
        return {"message": "Image uploaded successfully", "id": id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/delete_image", response_model=SuccessResponse)
@limiter.limit("10/hour")
def delete_image(request: Request, req: DeleteImageRequest, token: str = Depends(get_current_token)):
    try:
        entry = pastaloader.get(req.id)
        pastaloader.delete_image(req.id)
        log_action(request.client.host, "delete_image", f"Deleted image from '{entry['title']}' (id: {req.id})")
        return {"message": "Image deleted successfully", "id": req.id}
    except (ValueError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=408, detail=translate("request_timed_out"))
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/data_version", response_model=DataVersionResponse)
@limiter.limit("120/minute")
def data_version(request: Request):
    return {"version": pastaloader.version}

@app.get("/captcha_question", response_model=CaptchaQuestionResponse)
@limiter.limit("100/minute")
def captcha_question(request: Request):
    index = random.randint(0, len(CAPTCHA_QUESTIONS) - 1)
    question = CAPTCHA_QUESTIONS[index]["question"]
    return {"question": question, "index": index}

@app.post("/captcha_answer", response_model=CaptchaAnswerResponse)
@limiter.limit("10/30 minutes")
def captcha_answer(request: Request, req: CaptchaAnswerRequest):
    if req.index < 0 or req.index >= len(CAPTCHA_QUESTIONS):
        raise HTTPException(status_code=400, detail="Invalid index")
    answers = CAPTCHA_QUESTIONS[req.index]["answers"]
    if req.answer.lower() in [a.lower() for a in answers]:
        token = secrets.token_urlsafe(32)
        ip = request.client.host
        expiration = time.time() + 28 * 24 * 3600
        with token_lock:
            captcha_tokens[token] = {"ip": ip, "expiration": expiration}
        return {"token": token}
    else:
        raise HTTPException(status_code=400, detail="Incorrect answer")

@app.get("/verify_captcha")
@limiter.limit("100/minute")
def verify_captcha_endpoint(request: Request):
    if verify_captcha(request):
        return {"valid": True}
    else:
        raise HTTPException(status_code=400, detail="Invalid CAPTCHA")
