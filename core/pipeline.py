#!/usr/bin/env python3
"""
🎨 Creative Pipeline Module
Orchestrates the entire creative process from prompt to final output
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Any

import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from .llm_interface import LLMInterface
from .memory import MemorySystem
from .openfabric_integration import OpenfabricIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreativePipeline:
    """
    Main pipeline that orchestrates the creative process.
    Handles prompt enhancement, memory management, and output generation.
    """

    def __init__(self, memory_db_path: str = "creative_memory.db"):
        """
        Initialize the creative pipeline.

        Args:
            memory_db_path: Path to the SQLite database for memory storage
        """
        self.llm = LLMInterface()
        self.memory = MemorySystem(memory_db_path)
        self.openfabric = OpenfabricIntegration()
        self.output_dir = "outputs"
        self.openfabric_apps = []  # List of configured Openfabric app IDs

        # Ensure output directories exist
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)

        logger.info("Creative Pipeline initialized")

    def configure_openfabric_apps(self, app_ids: List[str]) -> Dict[str, Any]:
        """
        Configure Openfabric apps for enhanced generation.
        
        Args:
            app_ids: List of Openfabric app IDs to configure
            
        Returns:
            Configuration status
        """
        try:
            self.openfabric_apps = app_ids
            result = self.openfabric.configure_apps(app_ids)
            logger.info(f"Openfabric apps configured: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to configure Openfabric apps: {e}")
            return {"success": False, "error": str(e)}

    def create_from_prompt(self, prompt: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Create images and 3D models from a text prompt.

        Args:
            prompt: The user's text prompt
            user_id: Unique identifier for the user

        Returns:
            Dictionary with creation results and metadata
        """
        try:
            start_time = time.time()

            # Step 1: Analyze and enhance the prompt
            logger.info(f"Processing prompt: {prompt}")
            analysis = self.llm.analyze_prompt(prompt)
            enhanced_prompt = self.llm.enhance_prompt(prompt)

            # Step 2: Generate unique identifiers
            creation_id = str(uuid.uuid4())
            timestamp = int(time.time())

            # Step 3: Generate real image and 3D model
            image_path = self._generate_real_image(enhanced_prompt, creation_id)
            model_path = self._generate_real_3d_model(enhanced_prompt, creation_id)

            # Step 4: Save to memory
            memory_id = self.memory.save_creation(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                image_path=image_path,
                model_3d_path=model_path,
                metadata={
                    "creation_id": creation_id,
                    "timestamp": timestamp,
                    "processing_time": time.time() - start_time,
                    "analysis": analysis,
                    "user_id": user_id,
                    "openfabric_apps": self.openfabric_apps
                }
            )

            # Step 5: Return results
            result = {
                "success": True,
                "memory_id": memory_id,
                "creation_id": creation_id,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_path,
                "model_3d_path": model_path,
                "analysis": analysis,
                "processing_time": time.time() - start_time,
                "openfabric_apps": self.openfabric_apps
            }

            logger.info(f"Creation completed: {creation_id}")
            return result

        except Exception as e:
            logger.error(f"Creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt
            }

    def _generate_real_image(self, prompt: str, creation_id: str) -> str:
        """
        Generate a real image using Openfabric apps or Hugging Face API.
        Falls back to placeholder if APIs fail.
        """
        filename = f"image_{creation_id[:8]}.png"
        image_path = f"{self.output_dir}/images/{filename}"
        
        # Try Openfabric apps first
        if self.openfabric_apps:
            for app_id in self.openfabric_apps:
                try:
                    logger.info(f"🎨 Trying Openfabric app for image: {app_id}")
                    image_data = self.openfabric.generate_image(app_id, prompt)
                    if image_data:
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        logger.info(f"✅ Image generated with Openfabric app: {app_id}")
                        return image_path
                except Exception as e:
                    logger.warning(f"❌ Openfabric app {app_id} failed: {e}")
                    continue
        
        # Fallback to Hugging Face API
        token = os.getenv("HUGGINGFACE_API_TOKEN")

        if not token:
            logger.warning("⚠️ HUGGINGFACE_API_TOKEN not set. Falling back to placeholder image.")
            return self._generate_fallback_image(prompt, image_path)

        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt}

        try:
            logger.info("🎨 Generating image via Hugging Face API...")
            response = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.warning(f"API Error {response.status_code}: {response.text}")
                return self._generate_fallback_image(prompt, image_path)

            image = Image.open(BytesIO(response.content))
            image.save(image_path)
            logger.info(f"✅ Image saved to {image_path}")
            return image_path

        except Exception as e:
            logger.error(f"❌ Failed to generate image: {e}")
            return self._generate_fallback_image(prompt, image_path)

    def _generate_fallback_image(self, prompt: str, image_path: str) -> str:
        """
        Generate a placeholder PNG image with prompt text.
        """
        img = Image.new("RGB", (512, 512), color=0x496D89)  # RGB(73, 109, 137) as hex
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Prompt:\n{prompt}", fill=(255, 255, 255), font=font)
        img.save(image_path)
        logger.info(f"📎 Fallback image saved at {image_path}")
        return image_path

    def _generate_real_3d_model(self, prompt: str, creation_id: str) -> str:
        """
        Generate a real 3D model using Openfabric apps.
        Falls back to placeholder if APIs fail.
        """
        filename = f"model_{creation_id[:8]}.glb"
        model_path = f"{self.output_dir}/models/{filename}"
        
        # Try Openfabric apps for 3D generation
        if self.openfabric_apps:
            for app_id in self.openfabric_apps:
                try:
                    logger.info(f"🎲 Trying Openfabric app for 3D model: {app_id}")
                    model_data = self.openfabric.generate_3d_model(app_id, prompt)
                    if model_data:
                        with open(model_path, 'wb') as f:
                            f.write(model_data)
                        logger.info(f"✅ 3D model generated with Openfabric app: {app_id}")
                        return model_path
                except Exception as e:
                    logger.warning(f"❌ Openfabric app {app_id} failed for 3D: {e}")
                    continue
        
        # Fallback to placeholder 3D model
        return self._generate_fallback_3d_model(prompt, model_path)

    def _generate_fallback_3d_model(self, prompt: str, model_path: str) -> str:
        """
        Generate a placeholder 3D model file.
        In a real implementation, this would create a basic GLB file.
        """
        # Create a simple GLB file structure (minimal valid GLB)
        glb_header = b'glTF'  # Magic
        version = (2).to_bytes(4, 'little')  # Version 2
        length = (12).to_bytes(4, 'little')  # Total length
        json_length = (0).to_bytes(4, 'little')  # JSON chunk length
        json_type = (0x4E4F534A).to_bytes(4, 'little')  # JSON chunk type
        
        # Create minimal GLB content
        glb_content = glb_header + version + length + json_length + json_type
        
        with open(model_path, 'wb') as f:
            f.write(glb_content)
        
        logger.info(f"📎 Fallback 3D model saved at {model_path}")
        return model_path

    def search_creations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            results = self.memory.search_creations(query, limit=limit)
            logger.info(f"Search returned {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_recent_creations(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            results = self.memory.get_recent_creations(limit=limit)
            logger.info(f"Retrieved {len(results)} recent creations")
            return results
        except Exception as e:
            logger.error(f"Failed to get recent creations: {e}")
            return []

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            analysis = self.llm.analyze_prompt(prompt)
            validation = self.llm.validate_prompt(prompt)
            result = {
                **analysis,
                "validation": validation,
                "enhanced_suggestion": self.llm.enhance_prompt(prompt),
                "variations": self.llm.generate_variations(prompt, count=3)
            }
            logger.info(f"Prompt analysis completed for: {prompt}")
            return result
        except Exception as e:
            logger.error(f"Prompt analysis failed: {e}")
            return {"error": str(e), "original_prompt": prompt}

    def get_creation_stats(self) -> Dict[str, Any]:
        try:
            stats = self.memory.get_creation_stats()
            logger.info("Retrieved creation statistics")
            return stats
        except Exception as e:
            logger.error(f"Failed to get creation stats: {e}")
            return {"error": str(e)}

    def get_openfabric_status(self) -> Dict[str, Any]:
        """
        Get the status of configured Openfabric apps.
        
        Returns:
            Status information for Openfabric integration
        """
        try:
            status = self.openfabric.get_app_status()
            logger.info("Retrieved Openfabric app status")
            return status
        except Exception as e:
            logger.error(f"Failed to get Openfabric status: {e}")
            return {"error": str(e)}

    def batch_generate(self, prompts: List[str], user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Generate content for multiple prompts in batch.
        
        Args:
            prompts: List of prompts to process
            user_id: Unique identifier for the user
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch item {i+1}/{len(prompts)}: {prompt}")
            
            try:
                result = self.create_from_prompt(prompt, user_id)
                result["batch_index"] = i
                result["batch_total"] = len(prompts)
                results.append(result)
                
                # Small delay between requests to avoid overwhelming APIs
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Batch item {i+1} failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "original_prompt": prompt,
                    "batch_index": i,
                    "batch_total": len(prompts)
                })
        
        logger.info(f"Batch processing complete: {len([r for r in results if r['success']])}/{len(prompts)} successful")
        return results

    def generate_variations(self, base_prompt: str, count: int = 3, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Generate multiple variations of a base prompt.
        
        Args:
            base_prompt: The base prompt to vary
            count: Number of variations to generate
            user_id: Unique identifier for the user
            
        Returns:
            List of generation results for variations
        """
        # Generate prompt variations
        variations = self.llm.generate_variations(base_prompt, count)
        
        # Generate content for each variation
        results = []
        for i, variation in enumerate(variations):
            logger.info(f"Generating variation {i+1}/{count}: {variation}")
            
            try:
                result = self.create_from_prompt(variation, user_id)
                result["variation_index"] = i
                result["variation_total"] = count
                result["base_prompt"] = base_prompt
                results.append(result)
                
                # Small delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Variation {i+1} failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "original_prompt": variation,
                    "variation_index": i,
                    "variation_total": count,
                    "base_prompt": base_prompt
                })
        
        logger.info(f"Variation generation complete: {len([r for r in results if r['success']])}/{count} successful")
        return results