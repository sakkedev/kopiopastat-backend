# Running the Kopiopasta API

This is a FastAPI-based web server for managing kopiopasta entries stored in `kopiopasta.json`. It includes authentication for privileged operations, logging, and automatic daily backups.

## Prerequisites

- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

## Setup

1. Copy `.env.example` to `.env` and set your desired TOKENS value (e.g., a comma-separated list of strong passwords for authentication).
2. Ensure `kopiopasta.json` is in the same directory (or create an empty list `[]` if starting fresh).

## Running the Server

1. Run the server with uvicorn:
   ```
   uvicorn kopiopasta_api:app --reload
   ```
   By default, it runs on `http://127.0.0.1:8000`. To run on a different port, specify the port:
   ```
   uvicorn kopiopasta_api:app --reload --port 8080
   ```
   Replace `8080` with your desired port number.
2. The API will be available at the specified address (e.g., `http://127.0.0.1:8000`).
3. Interactive API docs at `http://127.0.0.1:PORT/docs` (replace PORT with the actual port).

## Production Deployment on Debian VPS with Nginx

To run both backend and frontend on the same VPS under `https://kopiopastat.org`:

1. **Install Nginx and Certbot:**
   ```
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx
   ```

2. **Obtain SSL Certificate:**
   ```
   sudo certbot --nginx -d kopiopastat.org
   ```

3. **Configure Nginx:**
   - Place the provided `nginx.conf` in `/etc/nginx/sites-available/kopiopastat.org`.
   - Enable the site: `sudo ln -s /etc/nginx/sites-available/kopiopastat.org /etc/nginx/sites-enabled/`
   - Remove default: `sudo rm /etc/nginx/sites-enabled/default`
   - Test config: `sudo nginx -t`
   - Reload: `sudo systemctl reload nginx`

4. **Run Backend with Systemd:**
   - Place `kopiopasta.service` in `/etc/systemd/system/`.
   - Enable and start: `sudo systemctl enable kopiopasta && sudo systemctl start kopiopasta`

5. **Build and Serve Frontend:**
   - Assuming frontend is a React app, build it: `npm run build`
   - Copy build files to `/var/www/kopiopastat.org/html/`
   - Nginx will serve static files and proxy API requests to the backend.

## Authentication

Privileged endpoints (delete, logout) require a Bearer token obtained via `/login`. Tokens expire after 7 days of inactivity and are IP-bound.

## CAPTCHA

CAPTCHA verification is required for editing and creating entries. CAPTCHA token is passed in the X-Captcha header.

## Logging and Backups

- Actions (edit, new, delete) are logged to `logs/actions_{timestamp}.log` with timestamps, IP, action, and details.
- Daily backups of `kopiopasta.json` are created in `backups/kopiopasta_{date}.json` at startup and every 24 hours.

## Endpoints

### Public Endpoints

- GET `/browse?start=0&end=10`: Browse entries in alphabetical order (Finnish alphabet), returns version and contents list with title, id, and latest content (limited to 30 per minute).
- GET `/pasta/{id}`: Get latest content for an entry, includes filename if image exists (null otherwise) (limited to 180 per minute).
- GET `/pasta/{normalized_title}`: Get latest content for an entry by its normalized title, includes filename if image exists (null otherwise) (limited to 180 per minute).
- GET `/get_by_order?order=0`: Get entry by its alphabetical order index (limited to 100 per minute).
- GET `/random`: Get a random entry, includes filename if image exists (null otherwise) (limited to 120 per minute).
- GET `/history?id=1`: Get full history of an entry (limited to 50 per minute).
- GET `/search?q=query`: Search for entries by title or content (prioritizes title matches, ignores capitalization and special characters, minimum 3 characters, returns up to 5 best matches) (limited