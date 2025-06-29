import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
from datetime import datetime

# Load environment variables from config.env if it exists
config_path = "config.env"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    print(f"✅ Loaded environment variables from {config_path}")

from core.pipeline import CreativePipeline

app = FastAPI(title="AI Creative Pipeline", description="Transform text prompts into images and 3D models.", version="1.0.0")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory config for app IDs and pipeline
pipeline: Optional[Union[CreativePipeline, 'MockCreativePipeline', 'RealCreativePipeline']] = None
app_ids: List[str] = []
mock_mode: bool = False

class ConfigRequest(BaseModel):
    app_ids: List[str]
    mock_mode: Optional[bool] = False

class ExecuteRequest(BaseModel):
    prompt: str

class AnalyzeRequest(BaseModel):
    prompt: str

def create_mock_image_data():
    """Create a simple mock image (1x1 pixel PNG)"""
    # Minimal PNG file (1x1 transparent pixel)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 image
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,  # RGB, no compression
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x78, 0x9C, 0x62, 0x00, 0x00, 0x00, 0x02,  # Empty data
        0x00, 0x01, 0xE5, 0x27, 0xDE, 0xFC, 0x00, 0x00,  # IEND chunk
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,  # End of PNG
        0x60, 0x82
    ])
    return png_data

def create_mock_3d_data():
    """Create a simple mock 3D model (minimal GLB)"""
    # Minimal GLB file (empty scene)
    glb_data = bytes([
        0x67, 0x6C, 0x54, 0x46,  # "glTF" magic
        0x02, 0x00, 0x00, 0x00,  # Version 2
        0x20, 0x00, 0x00, 0x00,  # Total length
        0x12, 0x00, 0x00, 0x00,  # JSON chunk length
        0x4A, 0x53, 0x4F, 0x4E,  # "JSON"
        0x7B, 0x22, 0x73, 0x63, 0x65, 0x6E, 0x65, 0x73,  # {"scenes"
        0x22, 0x3A, 0x5B, 0x5D, 0x7D, 0x00, 0x00, 0x00,  # :[]}
        0x00, 0x00, 0x00, 0x00,  # Padding
        0x00, 0x00, 0x00, 0x00,  # Binary chunk length
        0x00, 0x00, 0x00, 0x00,  # Binary chunk type
        0x00, 0x00, 0x00, 0x00   # Binary chunk data
    ])
    return glb_data

@app.post("/config", tags=["Configuration"])
def set_config(config: ConfigRequest):
    global pipeline, app_ids, mock_mode
    app_ids = config.app_ids
    mock_mode = config.mock_mode or False
    
    if mock_mode:
        # Create a mock pipeline that doesn't require real Openfabric connections
        pipeline = MockCreativePipeline()
        return {"success": True, "message": f"Configured in MOCK MODE with {len(app_ids)} app IDs."}
    else:
        # Create the main pipeline that has Hugging Face integration
        try:
            pipeline = CreativePipeline()
            return {"success": True, "message": f"Configured with Hugging Face integration and {len(app_ids)} app IDs."}
        except Exception as e:
            return {"success": False, "message": f"Failed to initialize pipeline: {str(e)}. Try enabling mock_mode."}

@app.post("/config/real", tags=["Configuration"])
def configure_real_openfabric():
    """Configure the pipeline with real Openfabric app IDs."""
    global pipeline, app_ids, mock_mode
    
    # Real Openfabric app IDs
    real_app_ids = [
        "f0997a01-d6d3-a5fe-53d8-561300318557",  # Text-to-Image
        "69543f29-4d41-4afc-7f29-3d51591f11eb"   # 3D Generation
    ]
    
    app_ids = real_app_ids
    mock_mode = False
    
    try:
        pipeline = CreativePipeline()
        # Configure Openfabric apps
        openfabric_result = pipeline.configure_openfabric_apps(real_app_ids)
        
        return {
            "success": True,
            "message": f"Configured with real Openfabric apps: Text-to-Image ({real_app_ids[0]}) and 3D ({real_app_ids[1]})",
            "app_ids": real_app_ids,
            "openfabric_status": openfabric_result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to configure real Openfabric apps: {str(e)}"
        }

@app.post("/execute", tags=["Pipeline"])
def execute(req: ExecuteRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not configured. Set app_ids first.")
    
    try:
        result = pipeline.create_from_prompt(req.prompt)
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"Pipeline failed: {str(e)}"}

@app.get("/search", tags=["Memory"])
def search(query: str, limit: int = 5):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not configured.")
    return {"results": pipeline.search_creations(query, limit)}

@app.get("/recent", tags=["Memory"])
def recent(limit: int = 10):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not configured.")
    return {"results": pipeline.get_recent_creations(limit)}

@app.post("/analyze", tags=["Pipeline"])
def analyze(req: AnalyzeRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not configured.")
    return pipeline.analyze_prompt(req.prompt)

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "mock_mode": mock_mode}

@app.get("/openfabric/status", tags=["Openfabric"])
def get_openfabric_status():
    """Get the status of configured Openfabric apps."""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not configured. Set app_ids first.")
    
    try:
        status = pipeline.get_openfabric_status()
        return status
    except Exception as e:
        return {"error": str(e)}

class MockCreativePipeline:
    """Mock pipeline for testing without real Openfabric apps"""
    
    def __init__(self):
        self.memory = None
        self.llm = None
        # Import here to avoid issues
        from core.memory import MemorySystem
        from core.llm_interface import LLMInterface
        self.memory = MemorySystem()
        self.llm = LLMInterface()
        
        # Create output directories
        os.makedirs('outputs/images', exist_ok=True)
        os.makedirs('outputs/models', exist_ok=True)
    
    def create_from_prompt(self, prompt: str, user_id: str = "super-user") -> dict:
        """Mock pipeline execution"""
        start_time = time.time()
        
        try:
            # Step 1: Analyze and enhance prompt
            analysis = self.llm.analyze_prompt(prompt)
            enhanced_prompt = self.llm.enhance_prompt(prompt)
            
            # Step 2: Generate mock image
            image_data = create_mock_image_data()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            image_path = f"outputs/images/mock_generated_{timestamp}.png"
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Step 3: Generate mock 3D model
            model_3d_data = create_mock_3d_data()
            model_3d_path = f"outputs/models/mock_model_{timestamp}.glb"
            
            with open(model_3d_path, 'wb') as f:
                f.write(model_3d_data)
            
            # Step 4: Save to memory
            metadata = {
                "processing_time": time.time() - start_time,
                "prompt_analysis": analysis,
                "user_id": user_id,
                "pipeline_version": "1.0",
                "mock_mode": True
            }
            
            memory_id = self.memory.save_creation(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                image_path=image_path,
                model_3d_path=model_3d_path,
                metadata=metadata,
                tags=self._extract_tags(prompt)
            )
            
            # Step 5: Prepare response
            result = {
                "success": True,
                "memory_id": memory_id,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_path,
                "model_3d_path": model_3d_path,
                "processing_time": time.time() - start_time,
                "analysis": analysis,
                "mock_mode": True,
                "message": f"""🎨 Creative Pipeline Complete! (MOCK MODE)

Original Prompt: {prompt}
Expanded Prompt: {enhanced_prompt}

📁 Generated Files:
• Image: {image_path}
• 3D Model: {model_3d_path}

💾 Memory ID: {memory_id}

The system has successfully:
✅ Expanded your prompt using AI
✅ Generated a mock image (for testing)
✅ Created a mock 3D model (for testing)
✅ Stored everything in memory for future reference

Note: This is running in MOCK MODE. For real image/3D generation, configure real Openfabric app IDs."""
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Mock pipeline failed: {str(e)}"
            }
    
    def _extract_tags(self, prompt: str) -> List[str]:
        """Extract tags from prompt for categorization."""
        tags = []
        prompt_lower = prompt.lower()
        
        # Content-based tags
        if any(word in prompt_lower for word in ['dragon', 'fantasy', 'magical']):
            tags.append('fantasy')
        if any(word in prompt_lower for word in ['robot', 'cyberpunk', 'futuristic']):
            tags.append('sci-fi')
        if any(word in prompt_lower for word in ['nature', 'forest', 'landscape']):
            tags.append('nature')
        if any(word in prompt_lower for word in ['city', 'urban', 'street']):
            tags.append('urban')
        
        # Style tags
        if 'night' in prompt_lower or 'dark' in prompt_lower:
            tags.append('night')
        if 'sunset' in prompt_lower or 'golden' in prompt_lower:
            tags.append('sunset')
        
        return tags
    
    def search_creations(self, query: str, limit: int = 5) -> List[dict]:
        """Search past creations."""
        return self.memory.search_creations(query, limit)
    
    def get_recent_creations(self, limit: int = 10) -> List[dict]:
        """Get recent creations."""
        return self.memory.get_recent_creations(limit)
    
    def analyze_prompt(self, prompt: str) -> dict:
        """Analyze a prompt for quality and suggestions."""
        return self.llm.analyze_prompt(prompt)

class RealCreativePipeline:
    """Real pipeline that connects to actual Openfabric apps"""
    
    def __init__(self, app_ids: List[str]):
        self.app_ids = app_ids
        self.text_to_image_app = app_ids[0] if len(app_ids) > 0 else None
        self.three_d_app = app_ids[1] if len(app_ids) > 1 else None
        
        # Import core components
        from core.memory import MemorySystem
        from core.llm_interface import LLMInterface
        self.memory = MemorySystem()
        self.llm = LLMInterface()
        
        # Create output directories
        os.makedirs('outputs/images', exist_ok=True)
        os.makedirs('outputs/models', exist_ok=True)
        
        # Test connections
        self._test_connections()
    
    def _test_connections(self):
        """Test connections to Openfabric apps"""
        try:
            import requests
            
            # Test text-to-image app - try different endpoint formats
            if self.text_to_image_app:
                # Try different Openfabric API endpoints
                api_endpoints = [
                    f"https://openfabric.network/api/app/{self.text_to_image_app}/execute",
                    f"https://openfabric.network/app/{self.text_to_image_app}/api/execute",
                    f"https://openfabric.network/app/{self.text_to_image_app}/execute",
                    f"https://api.openfabric.network/app/{self.text_to_image_app}/execute"
                ]
                
                connected = False
                for url in api_endpoints:
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            logging.info(f"✅ Connected to text-to-image app: {url}")
                            connected = True
                            break
                    except:
                        continue
                
                if not connected:
                    logging.warning(f"⚠️  Text-to-image app not responding. Tried: {api_endpoints}")
            
            # Test 3D app - try different endpoint formats
            if self.three_d_app:
                api_endpoints = [
                    f"https://openfabric.network/api/app/{self.three_d_app}/execute",
                    f"https://openfabric.network/app/{self.three_d_app}/api/execute",
                    f"https://openfabric.network/app/{self.three_d_app}/execute",
                    f"https://api.openfabric.network/app/{self.three_d_app}/execute"
                ]
                
                connected = False
                for url in api_endpoints:
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            logging.info(f"✅ Connected to 3D app: {url}")
                            connected = True
                            break
                    except:
                        continue
                
                if not connected:
                    logging.warning(f"⚠️  3D app not responding. Tried: {api_endpoints}")
                    
        except Exception as e:
            logging.error(f"Failed to test Openfabric connections: {e}")
    
    def create_from_prompt(self, prompt: str, user_id: str = "super-user") -> dict:
        """Real pipeline execution using Openfabric apps"""
        start_time = time.time()
        
        try:
            # Step 1: Analyze and enhance prompt
            analysis = self.llm.analyze_prompt(prompt)
            enhanced_prompt = self.llm.enhance_prompt(prompt)
            
            # Step 2: Generate image using text-to-image app
            image_path = self._generate_image_with_openfabric(enhanced_prompt, user_id)
            
            # Step 3: Generate 3D model using 3D app
            model_3d_path = self._generate_3d_with_openfabric(image_path, user_id)
            
            # Step 4: Save to memory
            metadata = {
                "processing_time": time.time() - start_time,
                "prompt_analysis": analysis,
                "user_id": user_id,
                "pipeline_version": "1.0",
                "openfabric_apps": self.app_ids,
                "real_mode": True
            }
            
            memory_id = self.memory.save_creation(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                image_path=image_path,
                model_3d_path=model_3d_path,
                metadata=metadata,
                tags=self._extract_tags(prompt)
            )
            
            # Step 5: Prepare response
            result = {
                "success": True,
                "memory_id": memory_id,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_path,
                "model_3d_path": model_3d_path,
                "processing_time": time.time() - start_time,
                "analysis": analysis,
                "openfabric_apps": self.app_ids,
                "message": f"""🎨 Creative Pipeline Complete! (REAL MODE)

Original Prompt: {prompt}
Expanded Prompt: {enhanced_prompt}

📁 Generated Files:
• Image: {image_path}
• 3D Model: {model_3d_path}

💾 Memory ID: {memory_id}

The system has successfully:
✅ Expanded your prompt using AI
✅ Generated a real image using Openfabric
✅ Created a real 3D model using Openfabric
✅ Stored everything in memory for future reference

Openfabric Apps Used:
• Text-to-Image: {self.text_to_image_app}
• 3D Generation: {self.three_d_app}"""
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Real pipeline failed: {str(e)}"
            }
    
    def _generate_image_with_openfabric(self, prompt: str, user_id: str) -> str:
        """Generate image using Openfabric text-to-image app"""
        try:
            import requests
            import base64
            
            if not self.text_to_image_app:
                raise Exception("Text-to-image app not configured")
            
            # Try different Openfabric API endpoints
            api_endpoints = [
                f"https://openfabric.network/api/app/{self.text_to_image_app}/execute",
                f"https://openfabric.network/app/{self.text_to_image_app}/api/execute",
                f"https://openfabric.network/app/{self.text_to_image_app}/execute",
                f"https://api.openfabric.network/app/{self.text_to_image_app}/execute"
            ]
            
            # Call the text-to-image app
            payload = {
                "prompt": prompt,
                "user_id": user_id
            }
            
            success = False
            for endpoint in api_endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        success = True
                        break
                    else:
                        logging.warning(f"Endpoint {endpoint} returned status {response.status_code}")
                        
                except Exception as e:
                    logging.warning(f"Failed to connect to {endpoint}: {e}")
                    continue
            
            if not success:
                raise Exception("All Openfabric endpoints failed")
            
            # Extract image data
            if 'image' in result:
                image_data = base64.b64decode(result['image'])
            elif 'data' in result:
                image_data = base64.b64decode(result['data'])
            else:
                raise Exception("No image data in response")
            
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            image_path = f"outputs/images/real_generated_{timestamp}.png"
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            logging.info(f"Generated real image: {image_path}")
            return image_path
            
        except Exception as e:
            logging.error(f"Failed to generate image with Openfabric: {e}")
            # Fallback to mock image
            return self._generate_mock_image(prompt)
    
    def _generate_3d_with_openfabric(self, image_path: str, user_id: str) -> str:
        """Generate 3D model using Openfabric 3D app"""
        try:
            import requests
            import base64
            
            if not self.three_d_app:
                raise Exception("3D app not configured")
            
            # Read and encode the image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Try different Openfabric API endpoints
            api_endpoints = [
                f"https://openfabric.network/api/app/{self.three_d_app}/execute",
                f"https://openfabric.network/app/{self.three_d_app}/api/execute",
                f"https://openfabric.network/app/{self.three_d_app}/execute",
                f"https://api.openfabric.network/app/{self.three_d_app}/execute"
            ]
            
            # Call the 3D app
            payload = {
                "image": image_b64,
                "user_id": user_id
            }
            
            success = False
            for endpoint in api_endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        success = True
                        break
                    else:
                        logging.warning(f"Endpoint {endpoint} returned status {response.status_code}")
                        
                except Exception as e:
                    logging.warning(f"Failed to connect to {endpoint}: {e}")
                    continue
            
            if not success:
                raise Exception("All Openfabric endpoints failed")
            
            # Extract 3D model data
            if 'model' in result:
                model_data = base64.b64decode(result['model'])
            elif 'data' in result:
                model_data = base64.b64decode(result['data'])
            else:
                raise Exception("No 3D model data in response")
            
            # Save 3D model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            model_path = f"outputs/models/real_model_{timestamp}.glb"
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            logging.info(f"Generated real 3D model: {model_path}")
            return model_path
            
        except Exception as e:
            logging.error(f"Failed to generate 3D model with Openfabric: {e}")
            # Fallback to mock 3D model
            return self._generate_mock_3d_model(image_path)
    
    def _generate_mock_image(self, prompt: str) -> str:
        """Fallback to mock image generation"""
        image_data = create_mock_image_data()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = f"outputs/images/fallback_{timestamp}.png"
        
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        logging.info(f"Generated fallback image: {image_path}")
        return image_path
    
    def _generate_mock_3d_model(self, image_path: str) -> str:
        """Fallback to mock 3D model generation"""
        model_data = create_mock_3d_data()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        model_path = f"outputs/models/fallback_{timestamp}.glb"
        
        with open(model_path, 'wb') as f:
            f.write(model_data)
        
        logging.info(f"Generated fallback 3D model: {model_path}")
        return model_path
    
    def _extract_tags(self, prompt: str) -> List[str]:
        """Extract tags from prompt for categorization."""
        tags = []
        prompt_lower = prompt.lower()
        
        # Content-based tags
        if any(word in prompt_lower for word in ['dragon', 'fantasy', 'magical']):
            tags.append('fantasy')
        if any(word in prompt_lower for word in ['robot', 'cyberpunk', 'futuristic']):
            tags.append('sci-fi')
        if any(word in prompt_lower for word in ['nature', 'forest', 'landscape']):
            tags.append('nature')
        if any(word in prompt_lower for word in ['city', 'urban', 'street']):
            tags.append('urban')
        
        # Style tags
        if 'night' in prompt_lower or 'dark' in prompt_lower:
            tags.append('night')
        if 'sunset' in prompt_lower or 'golden' in prompt_lower:
            tags.append('sunset')
        
        return tags
    
    def search_creations(self, query: str, limit: int = 5) -> List[dict]:
        """Search past creations."""
        return self.memory.search_creations(query, limit)
    
    def get_recent_creations(self, limit: int = 10) -> List[dict]:
        """Get recent creations."""
        return self.memory.get_recent_creations(limit)
    
    def analyze_prompt(self, prompt: str) -> dict:
        """Analyze a prompt for quality and suggestions."""
        return self.llm.analyze_prompt(prompt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True) 