"""
ListinGenius API Backend
Video generation with Higgsfield API (CORRECT implementation)
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
HF_MODEL_ID = "kling-video/v2.1/pro/image-to-video"

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
    print(f"   Higgsfield Model: {HF_MODEL_ID}")
    yield
    print("👋 ListinGenius API Shutting down...")

app = FastAPI(
    title="ListinGenius API",
    description="AI-Powered Real Estate Video Generation",
    version="4.0.0",
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
# HIGGSFIELD VIDEO GENERATION - CORRECT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

def get_higgsfield_headers() -> dict:
    """Get correct Higgsfield authorization headers"""
    return {
        "Authorization": f"Key {HF_API_KEY}:{HF_API_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


async def submit_video_request(image_url: str, prompt: str, aspect_ratio: str) -> dict:
    """Submit a video generation request to Higgsfield"""
    
    if not HF_API_KEY or not HF_API_SECRET:
        return {"status": "not_configured", "error": "API keys not set"}
    
    # Map aspect ratio to Higgsfield format if needed
    # Higgsfield uses the image's aspect ratio, but we can add it to prompt
    aspect_prompts = {
        "9:16": "vertical format, portrait orientation",
        "1:1": "square format",
        "16:9": "horizontal format, landscape orientation, widescreen"
    }
    
    full_prompt = f"{prompt}, {aspect_prompts.get(aspect_ratio, '')}"
    
    url = f"{HF_BASE_URL}/{HF_MODEL_ID}"
    
    payload = {
        "image_url": image_url,
        "prompt": full_prompt,
        "duration": 5
    }
    
    headers = get_higgsfield_headers()
    
    print(f"      📤 Submitting to: {url}")
    print(f"      📤 Prompt: {full_prompt[:80]}...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"      📥 Status: {response.status_code}")
            
            if response.status_code in [200, 201, 202]:
                data = response.json()
                print(f"      📥 Response: {data}")
                return {
                    "status": "submitted",
                    "request_id": data.get("request_id"),
                    "status_url": data.get("status_url"),
                    "data": data
                }
            else:
                error_text = response.text[:500]
                print(f"      ❌ Error: {error_text}")
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {error_text}"
                }
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return {"status": "error", "error": str(e)}


async def poll_video_status(request_id: str, max_attempts: int = 120) -> dict:
    """Poll for video completion - waits up to 10 minutes"""
    
    status_url = f"{HF_BASE_URL}/requests/{request_id}/status"
    headers = get_higgsfield_headers()
    
    print(f"      ⏳ Polling status: {status_url}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_attempts):
            try:
                response = await client.get(status_url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()
                    
                    print(f"      📊 Attempt {attempt + 1}: {status}")
                    
                    if status == "completed":
                        video_url = data.get("video", {}).get("url")
                        if video_url:
                            return {"status": "completed", "url": video_url}
                        else:
                            return {"status": "completed", "url": None, "error": "No video URL in response"}
                    
                    elif status in ["failed", "error"]:
                        return {"status": "failed", "error": data.get("error", "Generation failed")}
                    
                    elif status == "nsfw":
                        return {"status": "failed", "error": "Content flagged as NSFW"}
                    
                    # Still processing - wait and retry
                    await asyncio.sleep(5)
                    
                else:
                    print(f"      ⚠️ Status check failed: {response.status_code}")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"      ⚠️ Poll error: {e}")
                await asyncio.sleep(5)
    
    return {"status": "timeout", "error": "Video generation timed out"}


async def generate_video_higgsfield(image_url: str, prompt: str, aspect_ratio: str) -> dict:
    """Full video generation flow: submit + poll"""
    
    # Step 1: Submit the request
    submit_result = await submit_video_request(image_url, prompt, aspect_ratio)
    
    if submit_result.get("status") != "submitted":
        return submit_result
    
    request_id = submit_result.get("request_id")
    if not request_id:
        return {"status": "error", "error": "No request_id returned"}
    
    # Step 2: Poll for completion
    result = await poll_video_status(request_id)
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

Write 2-3 engaging paragraphs (150-200 words):
- Hook them in the first sentence
- Paint a picture of the lifestyle
- Highlight unique features naturally
- End with urgency/call to action

Write ONLY the description. No headers, labels, or quotes."""

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
            else:
                print(f"Claude error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Claude API error: {e}")
    
    return generate_fallback_description(listing)


def generate_fallback_description(listing: ListingDetails) -> str:
    return f"""Welcome to {listing.address} – a stunning {listing.property_type.lower()} that perfectly blends comfort with sophistication. Priced at ${listing.price:,.0f}, this exceptional {listing.bedrooms}-bedroom, {listing.bathrooms}-bathroom residence offers {listing.sqft or 'generous'} square feet of thoughtfully designed living space.

Step inside to discover an open-concept layout bathed in natural light, featuring premium finishes throughout. The gourmet kitchen is a chef's dream, while the luxurious primary suite provides a private retreat.

Located in a highly sought-after neighborhood, this home represents an incredible opportunity for discerning buyers. Schedule your private showing today – properties like this don't last long!"""


async def generate_social_posts_with_claude(listing: ListingDetails) -> dict:
    """Generate social media posts using Claude"""
    
    default_posts = generate_fallback_social_posts(listing)
    
    if not ANTHROPIC_API_KEY:
        return default_posts
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = f"""Create social media posts for this real estate listing:

Property: {listing.address}
Price: ${listing.price:,.0f}
Specs: {listing.bedrooms} BD / {listing.bathrooms} BA / {listing.sqft or 'N/A'} SF
Type: {listing.property_type}

Return ONLY valid JSON:
{{"instagram": "engaging post with emojis and 5-8 hashtags", "facebook": "longer post with call to action", "tiktok": "short punchy caption with hashtags", "twitter": "compelling tweet under 280 chars", "youtube": "video description with keywords"}}

JSON only:"""

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
        "instagram": f"✨ JUST LISTED ✨\n\n📍 {addr}\n💰 {price}\n🏠 {specs}\n\nYour dream home awaits!\n\n#JustListed #RealEstate #DreamHome #HomeForSale #LuxuryListing",
        "facebook": f"🏠 NEW LISTING ALERT!\n\n{addr}\n{price} | {specs}\n\nContact me today for a private showing!",
        "tiktok": f"POV: Your dream home 🏡✨ {price} | {specs} #realestate #housetour #dreamhome #justlisted #fyp",
        "twitter": f"🏡 Just Listed: {addr}\n{price} | {specs}\nDM for details 📩",
        "youtube": f"{addr} | Home Tour | {price} | {listing.bedrooms} Bed {listing.bathrooms} Bath"
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
                    # Convert to direct download URL
                    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    
    raise Exception("Failed to upload image")

# ═══════════════════════════════════════════════════════════════
# VIDEO PROMPT
# ═══════════════════════════════════════════════════════════════

def build_video_prompt(listing: ListingDetails) -> str:
    """Build optimized prompt for video generation"""
    
    style_prompts = {
        "cinematic": "Smooth cinematic camera pan, professional real estate showcase, warm golden hour lighting, high-end property tour",
        "luxury": "Elegant slow camera movement, luxury real estate presentation, sophisticated lighting, premium property reveal",
        "modern": "Clean contemporary camera sweep, modern architectural focus, minimalist aesthetic, sleek property tour",
        "cozy": "Gentle floating camera movement, warm inviting atmosphere, soft natural lighting, welcoming home tour",
        "dramatic": "Bold dynamic camera sweep, dramatic lighting, striking architectural reveal, impressive property showcase"
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
                print(f"   ✅ Image uploaded: {image_url}")
                
                prompt = build_video_prompt(listing)
                
                # Generate vertical video only (for TikTok, Instagram, Facebook)
                job["message"] = "Generating video... This may take 1-2 minutes."
                job["progress"] = 40
                
                print(f"\n   🎥 Generating vertical video (9:16)...")
                result = await generate_video_higgsfield(image_url, prompt, "9:16")
                videos["vertical"] = result
                
                # Set other formats to reference the vertical video
                videos["square"] = {"status": "not_generated", "url": None, "message": "Use vertical video"}
                videos["landscape"] = {"status": "not_generated", "url": None, "message": "Use vertical video"}
                
                status_icon = "✅" if result.get("status") == "completed" and result.get("url") else "❌"
                print(f"   {status_icon} vertical: {result.get('status')} - {result.get('url', result.get('error', 'No URL'))[:60] if result.get('url') else result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"   ❌ Video error: {e}")
                import traceback
                traceback.print_exc()
                for fmt in ["vertical", "square", "landscape"]:
                    if fmt not in videos:
                        videos[fmt] = {"status": "failed", "error": str(e), "url": None}
        else:
            reason = "API keys not configured" if not (HF_API_KEY and HF_API_SECRET) else "No images provided"
            for fmt in ["vertical", "square", "landscape"]:
                videos[fmt] = {"status": "not_configured", "url": None, "error": reason}
        
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
        import traceback
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "ListinGenius API",
        "version": "4.0.0",
        "status": "running",
        "capabilities": {
            "video_generation": bool(HF_API_KEY and HF_API_SECRET),
            "ai_descriptions": bool(ANTHROPIC_API_KEY),
            "ai_social_posts": bool(ANTHROPIC_API_KEY)
        },
        "video_model": HF_MODEL_ID
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "anthropic": "connected" if ANTHROPIC_API_KEY else "not configured",
            "higgsfield": "connected" if (HF_API_KEY and HF_API_SECRET) else "not configured"
        }
    }

@app.post("/api/jobs/create")
async def create_job(
    background_tasks: BackgroundTasks,
    listing_json: str = Form(...),
    photos: List[UploadFile] = File(...)
):
    """Create a video generation job"""
    
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
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/demo/generate")
async def demo_generate(listing: ListingDetails):
    """Demo endpoint - generates text only"""
    description = await generate_description_with_claude(listing)
    social_posts = await generate_social_posts_with_claude(listing)
    
    return {
        "description": description,
        "social_posts": social_posts,
        "videos": {
            "vertical": {"status": "demo", "url": None},
            "square": {"status": "demo", "url": None},
            "landscape": {"status": "demo", "url": None}
        }
    }

# Test endpoint to verify Higgsfield connection
@app.get("/api/test/higgsfield")
async def test_higgsfield():
    """Test Higgsfield API connection"""
    if not HF_API_KEY or not HF_API_SECRET:
        return {"status": "error", "message": "API keys not configured"}
    
    # Try a simple status check
    headers = get_higgsfield_headers()
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Just check if we can reach the platform
            response = await client.get(
                f"{HF_BASE_URL}/requests/test-invalid-id/status",
                headers=headers
            )
            # 404 means auth worked but request doesn't exist (expected)
            # 401 means auth failed
            if response.status_code == 404:
                return {"status": "ok", "message": "Higgsfield API connection successful"}
            elif response.status_code == 401:
                return {"status": "error", "message": "Invalid API credentials"}
            else:
                return {"status": "unknown", "code": response.status_code, "response": response.text[:200]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
