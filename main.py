"""
ListinGenius API Backend
Video generation with Higgsfield Cloud API (direct HTTP)
Text generation with Claude AI
"""

import os
import json
import uuid
import asyncio
import base64
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_API_SECRET = os.environ.get("HF_API_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job storage
jobs = {}

# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ListinGenius API Starting...")
    print(f"   Anthropic API: {'✅ Configured' if ANTHROPIC_API_KEY else '❌ Missing'}")
    print(f"   Higgsfield API: {'✅ Configured' if HF_API_KEY else '❌ Missing'}")
    yield
    print("👋 ListinGenius API Shutting down...")

app = FastAPI(
    title="ListinGenius API",
    description="AI-Powered Real Estate Video Generation",
    version="3.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class ListingDetails(BaseModel):
    address: str
    price: float
    bedrooms: int
    bathrooms: float
    sqft: Optional[int] = None
    property_type: str = "Single Family Home"
    features: Optional[List[str]] = None
    style: str = "cinematic"
    director_notes: Optional[str] = None

# ═══════════════════════════════════════════════════════════════
# HIGGSFIELD VIDEO GENERATION - Direct HTTP with correct auth
# ═══════════════════════════════════════════════════════════════

async def generate_video_higgsfield(image_url: str, prompt: str, aspect_ratio: str) -> dict:
    """Generate video using Higgsfield Cloud API with correct authentication"""
    
    if not HF_API_KEY or not HF_API_SECRET:
        return {"status": "not_configured", "url": None}
    
    # Create credentials in the format expected by Higgsfield
    credentials = f"{HF_API_KEY}:{HF_API_SECRET}"
    
    # Try different auth methods
    headers_options = [
        # Option 1: Bearer token with credentials
        {"Authorization": f"Bearer {credentials}", "Content-Type": "application/json"},
        # Option 2: Basic auth
        {"Authorization": f"Basic {base64.b64encode(credentials.encode()).decode()}", "Content-Type": "application/json"},
        # Option 3: X-API-Key header
        {"X-API-Key": HF_API_KEY, "X-API-Secret": HF_API_SECRET, "Content-Type": "application/json"},
    ]
    
    payload = {
        "image_url": image_url,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": 5
    }
    
    # Also try different payload formats
    payload_v2 = {
        "input": {
            "image_url": image_url,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": 5
        }
    }
    
    # Try different endpoints
    endpoints = [
        "https://cloud.higgsfield.ai/api/v1/kling/image-to-video",
        "https://api.higgsfield.ai/v1/kling/image-to-video",
        "https://cloud.higgsfield.ai/kling/v1.6/pro/image-to-video",
    ]
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        for endpoint in endpoints:
            for headers in headers_options:
                for p in [payload, payload_v2]:
                    try:
                        print(f"      Trying: {endpoint}")
                        print(f"      Auth: {list(headers.keys())}")
                        
                        response = await client.post(endpoint, headers=headers, json=p)
                        
                        print(f"      Status: {response.status_code}")
                        
                        if response.status_code == 200 or response.status_code == 201:
                            data = response.json()
                            print(f"      Response: {data}")
                            
                            # Check for video URL or request ID
                            if data.get("video", {}).get("url"):
                                return {"status": "completed", "url": data["video"]["url"]}
                            
                            if data.get("request_id"):
                                # Poll for completion
                                request_id = data["request_id"]
                                video_url = await poll_higgsfield_job(client, request_id, headers)
                                if video_url:
                                    return {"status": "completed", "url": video_url}
                        
                        elif response.status_code != 401:
                            # Not auth error, log it
                            print(f"      Response: {response.text[:200]}")
                            
                    except Exception as e:
                        print(f"      Error: {e}")
                        continue
    
    return {"status": "failed", "error": "All auth methods failed", "url": None}


async def poll_higgsfield_job(client: httpx.AsyncClient, request_id: str, headers: dict) -> Optional[str]:
    """Poll for job completion"""
    status_endpoints = [
        f"https://cloud.higgsfield.ai/api/v1/requests/{request_id}/status",
        f"https://api.higgsfield.ai/v1/requests/{request_id}/status",
    ]
    
    for attempt in range(60):  # 5 minutes max
        await asyncio.sleep(5)
        
        for endpoint in status_endpoints:
            try:
                response = await client.get(endpoint, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()
                    
                    if status == "completed":
                        return data.get("video", {}).get("url")
                    elif status in ["failed", "error"]:
                        return None
            except:
                continue
    
    return None

# ═══════════════════════════════════════════════════════════════
# CLAUDE AI - Descriptions & Social Posts
# ═══════════════════════════════════════════════════════════════

async def generate_description_with_claude(listing: ListingDetails) -> str:
    """Generate compelling property description using Claude"""
    
    if not ANTHROPIC_API_KEY:
        return generate_fallback_description(listing)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = f"""You are a luxury real estate copywriter. Write a compelling listing description for:

Property: {listing.property_type}
Address: {listing.address}
Price: ${listing.price:,.0f}
Bedrooms: {listing.bedrooms} | Bathrooms: {listing.bathrooms}
Square Feet: {listing.sqft or 'Not specified'}
Key Features: {', '.join(listing.features) if listing.features else 'Modern updates throughout'}
Style/Vibe: {listing.style}
{f"Special Notes: {listing.director_notes}" if listing.director_notes else ""}

Write 2-3 engaging paragraphs (150-200 words). Write ONLY the description."""

        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 600,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            print(f"Claude API error: {e}")
    
    return generate_fallback_description(listing)


def generate_fallback_description(listing: ListingDetails) -> str:
    return f"""Welcome to {listing.address} – a stunning {listing.property_type.lower()} priced at ${listing.price:,.0f}. This exceptional {listing.bedrooms}-bedroom, {listing.bathrooms}-bathroom residence offers thoughtfully designed living space.

Step inside to discover an open-concept layout bathed in natural light, featuring premium finishes throughout. The gourmet kitchen is a chef's dream, while the luxurious primary suite provides a private retreat.

Located in a highly sought-after neighborhood, this home represents an incredible opportunity. Schedule your private showing today!"""


async def generate_social_posts_with_claude(listing: ListingDetails) -> dict:
    """Generate social media posts using Claude"""
    
    default_posts = generate_fallback_social_posts(listing)
    
    if not ANTHROPIC_API_KEY:
        return default_posts
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = f"""Create social media posts for this real estate listing:

Property: {listing.address}
Price: ${listing.price:,.0f}
Specs: {listing.bedrooms} BD / {listing.bathrooms} BA

Return ONLY valid JSON:
{{"instagram": "post with emojis and hashtags", "facebook": "longer post", "tiktok": "short caption", "twitter": "tweet under 280 chars", "youtube": "video description"}}"""

        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1200,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data["content"][0]["text"].strip()
                import re
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            print(f"Social posts error: {e}")
    
    return default_posts


def generate_fallback_social_posts(listing: ListingDetails) -> dict:
    addr = listing.address
    price = f"${listing.price:,.0f}"
    specs = f"{listing.bedrooms}BD/{listing.bathrooms}BA"
    
    return {
        "instagram": f"✨ JUST LISTED ✨\n\n📍 {addr}\n💰 {price}\n🏠 {specs}\n\n#JustListed #RealEstate #DreamHome",
        "facebook": f"🏠 NEW LISTING!\n\n{addr}\n{price} | {specs}\n\nContact me for a showing!",
        "tiktok": f"Dream home alert 🏡 {price} | {specs} #realestate #fyp",
        "twitter": f"🏡 Just Listed: {addr}\n{price} | {specs}",
        "youtube": f"{addr} | Home Tour | {price}"
    }

# ═══════════════════════════════════════════════════════════════
# IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════════

async def upload_image_to_public_url(image_path: str) -> str:
    """Upload image and get a public URL"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                url = data.get("data", {}).get("url", "")
                if url:
                    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    
    raise Exception("Failed to upload image")

# ═══════════════════════════════════════════════════════════════
# VIDEO PROMPT
# ═══════════════════════════════════════════════════════════════

def build_video_prompt(listing: ListingDetails) -> str:
    style_prompts = {
        "cinematic": "Smooth cinematic camera movement, professional real estate showcase",
        "luxury": "Prestigious slow reveal, luxury real estate tour",
        "modern": "Clean contemporary movement, modern architectural focus",
        "cozy": "Warm gentle floating movement, inviting home atmosphere",
        "dramatic": "Bold dynamic camera sweep, dramatic lighting"
    }
    
    base = style_prompts.get(listing.style, style_prompts["cinematic"])
    prompt = f"{base}, showcasing a beautiful {listing.property_type.lower()}"
    
    if listing.director_notes:
        prompt += f", {listing.director_notes}"
    
    return prompt

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

async def process_listing_job(job_id: str, listing: ListingDetails, image_paths: List[str]):
    """Main video generation pipeline"""
    job = jobs[job_id]
    
    try:
        print(f"\n🎬 Starting job {job_id}")
        
        # Step 1: Generate description
        job["status"] = "processing"
        job["message"] = "Generating AI description..."
        job["progress"] = 5
        
        description = await generate_description_with_claude(listing)
        job["description"] = description
        job["progress"] = 15
        print(f"   ✅ Description generated")
        
        # Step 2: Generate social posts
        job["message"] = "Creating social media posts..."
        social_posts = await generate_social_posts_with_claude(listing)
        job["social_posts"] = social_posts
        job["progress"] = 25
        print(f"   ✅ Social posts generated")
        
        # Step 3: Generate videos
        videos = {}
        
        if HF_API_KEY and HF_API_SECRET and image_paths:
            try:
                job["message"] = "Uploading image..."
                job["progress"] = 30
                
                image_url = await upload_image_to_public_url(image_paths[0])
                print(f"   ✅ Image uploaded: {image_url[:50]}...")
                
                formats = [("vertical", "9:16"), ("square", "1:1"), ("landscape", "16:9")]
                prompt = build_video_prompt(listing)
                
                for i, (fmt, ar) in enumerate(formats):
                    job["message"] = f"Generating {fmt} video ({i+1}/3)..."
                    job["progress"] = 35 + (i * 20)
                    
                    print(f"   🎥 Generating {fmt} video...")
                    result = await generate_video_higgsfield(image_url, prompt, ar)
                    videos[fmt] = result
                    print(f"   {'✅' if result.get('url') else '❌'} {fmt}: {result.get('status')}")
                    
            except Exception as e:
                print(f"   ❌ Video error: {e}")
                for fmt in ["vertical", "square", "landscape"]:
                    if fmt not in videos:
                        videos[fmt] = {"status": "failed", "error": str(e), "url": None}
        else:
            for fmt in ["vertical", "square", "landscape"]:
                videos[fmt] = {"status": "not_configured", "url": None}
        
        job["videos"] = videos
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Your listing package is ready!"
        job["completed_at"] = datetime.now().isoformat()
        
        print(f"✅ Job {job_id} completed!")
        
    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["error"] = str(e)
        print(f"❌ Job {job_id} failed: {e}")

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "ListinGenius API",
        "version": "3.3.0",
        "status": "running",
        "capabilities": {
            "video_generation": bool(HF_API_KEY and HF_API_SECRET),
            "ai_descriptions": bool(ANTHROPIC_API_KEY),
            "ai_social_posts": bool(ANTHROPIC_API_KEY)
        }
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/jobs/create")
async def create_job(
    background_tasks: BackgroundTasks,
    listing_json: str = Form(...),
    photos: List[UploadFile] = File(...)
):
    listing_data = json.loads(listing_json)
    listing = ListingDetails(**listing_data)
    
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i, photo in enumerate(photos):
        ext = Path(photo.filename).suffix or ".jpg"
        file_path = job_dir / f"photo_{i}{ext}"
        content = await photo.read()
        with open(file_path, "wb") as f:
            f.write(content)
        image_paths.append(str(file_path))
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "videos": None,
        "description": None,
        "social_posts": None,
        "created_at": datetime.now().isoformat(),
        "listing": listing_data
    }
    
    background_tasks.add_task(process_listing_job, job_id, listing, image_paths)
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/demo/generate")
async def demo_generate(listing: ListingDetails):
    description = await generate_description_with_claude(listing)
    social_posts = await generate_social_posts_with_claude(listing)
    return {
        "description": description,
        "social_posts": social_posts,
        "videos": {"vertical": {"status": "demo"}, "square": {"status": "demo"}, "landscape": {"status": "demo"}}
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
