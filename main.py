"""
ListinGenius API Backend
Video generation with Higgsfield API - Kling 3.0
Text generation with Claude AI
"""

import os
import json
import uuid
import asyncio
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

# Higgsfield API Configuration
HF_BASE_URL = "https://platform.higgsfield.ai"
HF_MODEL_ID = "higgsfield-ai/dop/standard"  # Working model

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
    print(f"   Models to try: {HF_MODELS}")
    yield
    print("👋 ListinGenius API Shutting down...")

app = FastAPI(
    title="ListinGenius API",
    description="AI-Powered Real Estate Video Generation",
    version="5.0.0",
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
    duration: int = 5  # 5, 10, or 15 seconds
    multi_shot: bool = False  # Enable multi-shot mode

# ═══════════════════════════════════════════════════════════════
# HIGGSFIELD VIDEO GENERATION - Kling 3.0
# ═══════════════════════════════════════════════════════════════

def get_higgsfield_headers() -> dict:
    """Get correct Higgsfield authorization headers"""
    return {
        "Authorization": f"Key {HF_API_KEY}:{HF_API_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


async def submit_video_request(
    image_urls: List[str], 
    prompt: str, 
    duration: int = 5,
    multi_shot: bool = False,
    multi_prompts: List[str] = None
) -> dict:
    """Submit a video generation request to Higgsfield"""
    
    if not HF_API_KEY or not HF_API_SECRET:
        return {"status": "not_configured", "error": "API keys not set"}
    
    headers = get_higgsfield_headers()
    
    # Build payload
    payload = {
        "image_url": image_urls[0],
        "prompt": prompt,
        "duration": duration
    }
    
    url = f"{HF_BASE_URL}/{HF_MODEL_ID}"
    
    print(f"      📤 Submitting to: {url}")
    print(f"      📤 Duration: {duration}s")
    print(f"      📤 Prompt: {prompt[:80]}...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"      📥 Status: {response.status_code}")
            
            if response.status_code in [200, 201, 202]:
                data = response.json()
                print(f"      ✅ Request accepted!")
                return {
                    "status": "submitted",
                    "request_id": data.get("request_id"),
                    "data": data
                }
            else:
                error_text = response.text[:300]
                print(f"      ❌ Error: {error_text}")
                return {"status": "error", "error": f"HTTP {response.status_code}: {error_text}"}
                
    except Exception as e:
        print(f"      ❌ Exception: {e}")
        return {"status": "error", "error": str(e)}


async def poll_video_status(request_id: str, max_attempts: int = 180) -> dict:
    """Poll for video completion - waits up to 15 minutes for longer videos"""
    
    status_url = f"{HF_BASE_URL}/requests/{request_id}/status"
    headers = get_higgsfield_headers()
    
    print(f"      ⏳ Polling status for request: {request_id}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_attempts):
            try:
                response = await client.get(status_url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()
                    
                    if attempt % 12 == 0:  # Log every minute
                        print(f"      📊 Status check {attempt + 1}: {status}")
                    
                    if status == "completed":
                        video_url = data.get("video", {}).get("url")
                        if video_url:
                            print(f"      ✅ Video ready: {video_url[:60]}...")
                            return {"status": "completed", "url": video_url}
                        else:
                            return {"status": "completed", "url": None, "error": "No video URL"}
                    
                    elif status in ["failed", "error"]:
                        return {"status": "failed", "error": data.get("error", "Generation failed")}
                    
                    elif status == "nsfw":
                        return {"status": "failed", "error": "Content flagged as NSFW"}
                    
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"      ⚠️ Poll error: {e}")
                await asyncio.sleep(5)
    
    return {"status": "timeout", "error": "Video generation timed out"}


async def generate_video_higgsfield(
    image_urls: List[str], 
    prompt: str, 
    duration: int = 5,
    multi_shot: bool = False,
    multi_prompts: List[str] = None
) -> dict:
    """Full video generation flow: submit + poll"""
    
    # Submit the request
    submit_result = await submit_video_request(
        image_urls, prompt, duration, multi_shot, multi_prompts
    )
    
    if submit_result.get("status") != "submitted":
        return submit_result
    
    request_id = submit_result.get("request_id")
    if not request_id:
        return {"status": "error", "error": "No request_id returned"}
    
    # Poll for completion (longer timeout for longer videos)
    max_attempts = 60 + (duration * 10)  # More time for longer videos
    result = await poll_video_status(request_id, max_attempts)
    return result

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
    return f"""Welcome to {listing.address} – a stunning {listing.property_type.lower()} priced at ${listing.price:,.0f}. This {listing.bedrooms}-bedroom, {listing.bathrooms}-bathroom residence offers thoughtfully designed living space.

Step inside to discover an open-concept layout with premium finishes throughout. The gourmet kitchen and luxurious primary suite make this home truly special.

Located in a sought-after neighborhood, schedule your private showing today!"""


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
# VIDEO PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_video_prompt(listing: ListingDetails) -> str:
    """Build optimized prompt for video generation"""
    
    style_prompts = {
        "cinematic": "Smooth cinematic camera movement, professional real estate showcase, warm golden hour lighting",
        "luxury": "Elegant slow camera movement, luxury real estate presentation, sophisticated lighting",
        "modern": "Clean contemporary camera sweep, modern architectural focus, minimalist aesthetic",
        "cozy": "Gentle floating camera movement, warm inviting atmosphere, soft natural lighting",
        "dramatic": "Bold dynamic camera sweep, dramatic lighting, striking architectural reveal"
    }
    
    base = style_prompts.get(listing.style, style_prompts["cinematic"])
    prompt = f"{base}, showcasing a beautiful {listing.property_type.lower()}, vertical format 9:16"
    
    if listing.director_notes:
        prompt += f", {listing.director_notes}"
    
    return prompt


def build_multi_shot_prompts(listing: ListingDetails, num_shots: int) -> List[str]:
    """Build prompts for multi-shot video"""
    
    shot_templates = [
        "Exterior establishing shot, camera slowly pushes in toward the front entrance",
        "Interior living room reveal, smooth pan across the open floor plan",
        "Kitchen showcase, camera glides along countertops highlighting finishes",
        "Primary bedroom tour, gentle floating movement through the space",
        "Backyard/outdoor area, wide shot capturing the full outdoor living space"
    ]
    
    base_style = f"Professional real estate video, {listing.style} style, warm lighting"
    
    prompts = []
    for i in range(min(num_shots, len(shot_templates))):
        prompts.append(f"{base_style}, {shot_templates[i]}")
    
    return prompts

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

async def process_listing_job(job_id: str, listing: ListingDetails, image_paths: List[str]):
    """Main video generation pipeline"""
    job = jobs[job_id]
    
    try:
        print(f"\n🎬 Starting job {job_id}")
        print(f"   Duration: {listing.duration}s")
        print(f"   Multi-shot: {listing.multi_shot}")
        print(f"   Images: {len(image_paths)}")
        
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
        
        # Step 3: Generate video
        videos = {}
        
        if HF_API_KEY and HF_API_SECRET and image_paths:
            try:
                job["message"] = "Uploading images..."
                job["progress"] = 30
                
                # Upload all images (max 4)
                image_urls = []
                for i, path in enumerate(image_paths[:4]):  # Max 4 images
                    url = await upload_image_to_public_url(path)
                    image_urls.append(url)
                    print(f"   ✅ Image {i+1} uploaded: {url[:50]}...")
                
                prompt = build_video_prompt(listing)
                
                # Build multi-shot prompts if enabled
                multi_prompts = None
                if listing.multi_shot and len(image_urls) > 1:
                    multi_prompts = build_multi_shot_prompts(listing, len(image_urls))
                
                job["message"] = f"Generating {listing.duration}s video... This may take a few minutes."
                job["progress"] = 40
                
                print(f"\n   🎥 Generating video...")
                print(f"      Duration: {listing.duration}s")
                print(f"      Images: {len(image_urls)}")
                print(f"      Multi-shot: {listing.multi_shot}")
                
                result = await generate_video_higgsfield(
                    image_urls=image_urls,
                    prompt=prompt,
                    duration=listing.duration,
                    multi_shot=listing.multi_shot,
                    multi_prompts=multi_prompts
                )
                
                videos["vertical"] = result
                videos["square"] = {"status": "not_generated", "url": None}
                videos["landscape"] = {"status": "not_generated", "url": None}
                
                status_icon = "✅" if result.get("status") == "completed" and result.get("url") else "❌"
                print(f"   {status_icon} Video: {result.get('status')}")
                if result.get("model_used"):
                    print(f"      Model used: {result.get('model_used')}")
                    
            except Exception as e:
                print(f"   ❌ Video error: {e}")
                import traceback
                traceback.print_exc()
                videos["vertical"] = {"status": "failed", "error": str(e), "url": None}
                videos["square"] = {"status": "not_generated", "url": None}
                videos["landscape"] = {"status": "not_generated", "url": None}
        else:
            for fmt in ["vertical", "square", "landscape"]:
                videos[fmt] = {"status": "not_configured", "url": None}
        
        job["videos"] = videos
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Your listing package is ready!"
        job["completed_at"] = datetime.now().isoformat()
        
        print(f"\n✅ Job {job_id} completed!")
        
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
        "version": "5.0.0",
        "status": "running",
        "features": {
            "video_generation": bool(HF_API_KEY and HF_API_SECRET),
            "ai_descriptions": bool(ANTHROPIC_API_KEY),
            "duration_options": [5, 10, 15],
            "max_images": 4,
            "multi_shot": True
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
    """Create a video generation job"""
    
    # Validate max 4 photos
    if len(photos) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 photos allowed")
    
    listing_data = json.loads(listing_json)
    
    # Validate duration
    duration = listing_data.get("duration", 5)
    if duration not in [5, 10, 15]:
        listing_data["duration"] = 5
    
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
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
