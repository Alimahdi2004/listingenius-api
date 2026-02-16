"""
ListinGenius API Backend
Video generation with official Higgsfield SDK
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
    version="3.0.0",
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
# HIGGSFIELD CLIENT - Using Official API Format
# ═══════════════════════════════════════════════════════════════

class HiggsFieldClient:
    """Client for Higgsfield AI API using official SDK format"""
    
    BASE_URL = "https://platform.higgsfield.ai"
    
    def __init__(self):
        self.api_key = HF_API_KEY
        self.api_secret = HF_API_SECRET
        
    @property
    def headers(self):
        # Format credentials as expected by Higgsfield
        credentials = f"{self.api_key}:{self.api_secret}"
        return {
            "Authorization": f"Bearer {credentials}",
            "Content-Type": "application/json"
        }
    
    @property
    def is_configured(self):
        return bool(self.api_key and self.api_secret)
    
    async def generate_video(
        self,
        image_url: str,
        prompt: str,
        aspect_ratio: str = "9:16",
    ) -> dict:
        """Generate video using Higgsfield Kling model"""
        if not self.is_configured:
            raise Exception("Higgsfield API not configured")
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Use Kling image-to-video model
            payload = {
                "input": {
                    "image_url": image_url,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": 5
                }
            }
            
            print(f"   Starting video generation: {aspect_ratio}")
            print(f"   Using model: kling/v1.6/pro/image-to-video")
            
            # Submit the job
            response = await client.post(
                f"{self.BASE_URL}/kling/v1.6/pro/image-to-video",
                headers=self.headers,
                json=payload
            )
            
            print(f"   Response status: {response.status_code}")
            
            if response.status_code not in [200, 201, 202]:
                error_text = response.text
                print(f"   Error: {error_text}")
                raise Exception(f"API error {response.status_code}: {error_text}")
            
            result = response.json()
            request_id = result.get("request_id")
            
            if not request_id:
                print(f"   Full response: {result}")
                # Check if video is already ready
                if result.get("video", {}).get("url"):
                    return {
                        "status": "completed",
                        "video_url": result["video"]["url"]
                    }
                raise Exception("No request_id in response")
            
            print(f"   Request ID: {request_id}")
            
            # Poll for completion
            for attempt in range(120):
                await asyncio.sleep(5)
                
                status_response = await client.get(
                    f"{self.BASE_URL}/requests/{request_id}/status",
                    headers=self.headers
                )
                
                if status_response.status_code != 200:
                    print(f"   Status check failed: {status_response.status_code}")
                    continue
                
                status_data = status_response.json()
                status = status_data.get("status", "").lower()
                
                print(f"   Status: {status} (attempt {attempt + 1})")
                
                if status == "completed":
                    video_url = status_data.get("video", {}).get("url")
                    return {
                        "status": "completed",
                        "video_url": video_url,
                        "request_id": request_id
                    }
                elif status in ["failed", "error"]:
                    error = status_data.get("error", "Unknown error")
                    raise Exception(f"Generation failed: {error}")
            
            raise Exception("Generation timed out after 10 minutes")

# Initialize client
hf_client = HiggsFieldClient()

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
        except Exception as e:
            print(f"Claude API error: {e}")
    
    return generate_fallback_description(listing)


def generate_fallback_description(listing: ListingDetails) -> str:
    """Fallback description if Claude is unavailable"""
    return f"""Welcome to {listing.address} – a stunning {listing.property_type.lower()} that perfectly blends comfort with sophistication. Priced at ${listing.price:,.0f}, this exceptional {listing.bedrooms}-bedroom, {listing.bathrooms}-bathroom residence offers {listing.sqft or 'generous'} square feet of thoughtfully designed living space.

Step inside to discover an open-concept layout bathed in natural light, featuring premium finishes throughout. The gourmet kitchen is a chef's dream, while the luxurious primary suite provides a private retreat. {f"Notable features include {', '.join(listing.features[:3])}." if listing.features else ""}

Located in a highly sought-after neighborhood, this home represents an incredible opportunity for discerning buyers. Schedule your private showing today – properties like this don't last long!"""


async def generate_social_posts_with_claude(listing: ListingDetails) -> dict:
    """Generate social media posts using Claude"""
    
    default_posts = generate_fallback_social_posts(listing)
    
    if not ANTHROPIC_API_KEY:
        return default_posts
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = f"""Create social media posts for this real estate listing. Make them engaging and platform-appropriate.

Property: {listing.address}
Price: ${listing.price:,.0f}
Specs: {listing.bedrooms} BD / {listing.bathrooms} BA / {listing.sqft or 'N/A'} SF
Type: {listing.property_type}

Create posts for each platform. Return ONLY valid JSON:
{{
  "instagram": "engaging post with emojis and 5-8 relevant hashtags",
  "facebook": "longer post with call to action",
  "tiktok": "short punchy caption with hashtags",
  "twitter": "compelling tweet under 280 chars",
  "youtube": "video description with keywords"
}}

JSON only, no markdown or explanation:"""

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
    """Fallback social posts"""
    addr = listing.address
    price = f"${listing.price:,.0f}"
    specs = f"{listing.bedrooms}BD/{listing.bathrooms}BA"
    
    return {
        "instagram": f"✨ JUST LISTED ✨\n\n📍 {addr}\n💰 {price}\n🏠 {specs}\n\nYour dream home awaits! This stunning {listing.property_type.lower()} has everything you've been searching for.\n\n📲 Link in bio for full tour!\n\n#JustListed #RealEstate #DreamHome #HomeForSale #LuxuryListing",
        "facebook": f"🏠 NEW LISTING ALERT!\n\n{addr}\n{price} | {specs}\n\nI'm thrilled to present this beautiful {listing.property_type.lower()} to the market!\n\n📞 Contact me today for a private showing!",
        "tiktok": f"POV: You just found your dream home 🏡✨ {price} | {specs} #realestate #housetour #dreamhome #justlisted #fyp",
        "twitter": f"🏡 Just Listed: {addr}\n\n{price} | {specs}\n\nStunning {listing.property_type.lower()}!\n\nDM for details 📩",
        "youtube": f"{addr} | Full Home Tour | {price} | {listing.bedrooms} Bed {listing.bathrooms} Bath\n\nWelcome to this incredible {listing.property_type.lower()}!"
    }

# ═══════════════════════════════════════════════════════════════
# VIDEO PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_video_prompt(listing: ListingDetails, format_type: str) -> str:
    """Build optimized prompt for video generation"""
    
    style_prompts = {
        "cinematic": "Smooth cinematic camera movement, professional real estate showcase, warm golden hour lighting",
        "luxury": "Prestigious slow reveal, luxury real estate tour, sophisticated lighting",
        "modern": "Clean contemporary movement, modern architectural focus, minimalist aesthetic",
        "cozy": "Warm gentle floating movement, inviting home atmosphere, soft natural lighting",
        "dramatic": "Bold dynamic camera sweep, striking architectural reveal, dramatic lighting"
    }
    
    base_style = style_prompts.get(listing.style, style_prompts["cinematic"])
    
    prompt_parts = [
        base_style,
        f"showcasing a beautiful {listing.property_type.lower()}",
    ]
    
    if listing.director_notes:
        prompt_parts.append(listing.director_notes)
    
    return ", ".join(prompt_parts)

# ═══════════════════════════════════════════════════════════════
# IMAGE UPLOAD HELPER
# ═══════════════════════════════════════════════════════════════

async def upload_image_to_public_url(image_path: str) -> str:
    """Upload image and get a public URL for Higgsfield"""
    # For now, we'll use a free image hosting service
    # In production, use your own S3 bucket or similar
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = await client.post(
                "https://tmpfiles.org/api/v1/upload",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                # Convert tmpfiles URL to direct link
                url = data.get("data", {}).get("url", "")
                if url:
                    # Convert https://tmpfiles.org/123/image.jpg to https://tmpfiles.org/dl/123/image.jpg
                    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    
    raise Exception("Failed to upload image")

# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════

async def process_listing_job(job_id: str, listing: ListingDetails, image_paths: List[str]):
    """Main video generation pipeline"""
    job = jobs[job_id]
    
    try:
        print(f"\n🎬 Starting job {job_id}")
        
        # Step 1: Generate description (5-15%)
        job["status"] = "processing"
        job["message"] = "Generating AI description..."
        job["progress"] = 5
        
        description = await generate_description_with_claude(listing)
        job["description"] = description
        job["progress"] = 15
        print(f"   ✅ Description generated")
        
        # Step 2: Generate social posts (15-25%)
        job["message"] = "Creating social media posts..."
        social_posts = await generate_social_posts_with_claude(listing)
        job["social_posts"] = social_posts
        job["progress"] = 25
        print(f"   ✅ Social posts generated")
        
        # Step 3: Upload image and generate videos (25-95%)
        videos = {}
        
        if hf_client.is_configured and image_paths:
            try:
                job["message"] = "Uploading image..."
                job["progress"] = 30
                
                # Upload the first image
                image_url = await upload_image_to_public_url(image_paths[0])
                print(f"   ✅ Image uploaded: {image_url[:50]}...")
                
                # Generate videos in different formats
                formats = [
                    ("vertical", "9:16"),
                    ("square", "1:1"),
                    ("landscape", "16:9")
                ]
                
                progress_per_video = 20
                
                for i, (format_name, aspect_ratio) in enumerate(formats):
                    job["message"] = f"Generating {format_name} video ({i+1}/3)..."
                    
                    prompt = build_video_prompt(listing, format_name)
                    print(f"   🎥 Generating {format_name} video...")
                    
                    try:
                        result = await hf_client.generate_video(
                            image_url=image_url,
                            prompt=prompt,
                            aspect_ratio=aspect_ratio
                        )
                        videos[format_name] = {
                            "url": result.get("video_url"),
                            "status": "completed"
                        }
                        print(f"   ✅ {format_name} video completed")
                    except Exception as e:
                        print(f"   ❌ {format_name} video failed: {e}")
                        videos[format_name] = {
                            "url": None,
                            "status": "failed",
                            "error": str(e)
                        }
                    
                    job["progress"] = 35 + (progress_per_video * (i + 1))
                    
            except Exception as e:
                print(f"   ❌ Video generation error: {e}")
                for fmt in ["vertical", "square", "landscape"]:
                    videos[fmt] = {"status": "failed", "error": str(e), "url": None}
        else:
            # No video API configured
            for fmt in ["vertical", "square", "landscape"]:
                videos[fmt] = {"status": "not_configured", "url": None}
        
        # Complete
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
        "version": "3.0.0",
        "status": "running",
        "capabilities": {
            "video_generation": hf_client.is_configured,
            "ai_descriptions": bool(ANTHROPIC_API_KEY),
            "ai_social_posts": bool(ANTHROPIC_API_KEY)
        }
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "anthropic": "connected" if ANTHROPIC_API_KEY else "not configured",
            "higgsfield": "connected" if hf_client.is_configured else "not configured"
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
        "message": "Job queued, starting soon...",
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
    """Get job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/demo/generate")
async def demo_generate(listing: ListingDetails):
    """Demo endpoint - text generation only"""
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

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
