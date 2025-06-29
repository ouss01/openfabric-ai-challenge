#!/usr/bin/env python3
"""
🧠 LLM Interface Module
Provides intelligent prompt enhancement and analysis using template-based expansion
"""

import json
import logging
import random
import re
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from config.env
load_dotenv("config.env")

class LLMInterface:
    """
    LLM Interface provides intelligent prompt enhancement and analysis.
    Uses template-based expansion for consistent, high-quality results.
    """

    def __init__(self, model_path: Optional[str] = None, use_hf_api: bool = True):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path to local LLM model (.gguf file)
            use_hf_api: Whether to use Hugging Face API as fallback
        """
        self.model_path = model_path
        self.use_hf_api = use_hf_api
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to load local model if path provided
        if model_path and os.path.exists(model_path):
            self._load_local_model()
        elif use_hf_api:
            logger.info("Using Hugging Face API for LLM processing")
        else:
            logger.warning("No local model or API available, using template expansion")

    def _load_local_model(self):
        """Load local model using transformers."""
        try:
            logger.info(f"Loading local model from {self.model_path}")
            
            # For GGUF files, we need to use a different approach
            if self.model_path and self.model_path.endswith('.gguf'):
                logger.warning("GGUF files require llama-cpp-python. Using template expansion instead.")
                return
            
            # For regular Hugging Face models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.tokenizer = None
            self.model = None

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt for quality, complexity, and suggestions.
        
        Args:
            prompt: The input prompt to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if self.model and self.tokenizer:
            return self._analyze_with_local_model(prompt)
        elif self.use_hf_api:
            return self._analyze_with_hf_api(prompt)
        else:
            return self._analyze_with_template(prompt)

    def _analyze_with_local_model(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using local model."""
        try:
            if not self.model or not self.tokenizer:
                return self._analyze_with_template(prompt)
                
            system_prompt = """Analyze this creative prompt and extract key information in JSON format:
            - subject: main subject or focus
            - style: artistic style mentioned
            - mood: emotional tone
            - colors: color scheme or palette
            - lighting: lighting conditions
            - composition: composition style
            - details: specific details mentioned"""
            
            full_prompt = f"{system_prompt}\n\nPrompt: {prompt}\n\nAnalysis:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to extract JSON from response
            if "Analysis:" in response:
                analysis_text = response.split("Analysis:")[-1].strip()
                try:
                    return json.loads(analysis_text)
                except:
                    pass
            
            return self._analyze_with_template(prompt)
            
        except Exception as e:
            logger.error(f"Local model analysis failed: {e}")
            return self._analyze_with_template(prompt)

    def _analyze_with_hf_api(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using Hugging Face API."""
        try:
            api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not api_token:
                return self._analyze_with_template(prompt)
            
            model_id = "microsoft/DialoGPT-medium"
            headers = {"Authorization": f"Bearer {api_token}"}
            
            system_prompt = "Analyze this creative prompt and extract key information:"
            full_prompt = f"{system_prompt} {prompt}"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.3,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    analysis = result[0].get("generated_text", "")
                    # Try to parse as JSON or extract key info
                    try:
                        return json.loads(analysis)
                    except:
                        pass
            
            return self._analyze_with_template(prompt)
            
        except Exception as e:
            logger.error(f"Hugging Face API analysis failed: {e}")
            return self._analyze_with_template(prompt)

    def _analyze_with_template(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using template-based approach."""
        # Simple keyword-based analysis
        prompt_lower = prompt.lower()
        
        analysis = {
            "subject": prompt,
            "style": "realistic",
            "mood": "neutral",
            "colors": "natural",
            "lighting": "natural",
            "composition": "balanced",
            "details": "standard",
            "word_count": len(prompt.split()),
            "estimated_quality": "good",
            "complexity": "medium",
            "suggestions": []
        }
        
        # Detect style keywords
        if any(word in prompt_lower for word in ["cartoon", "anime", "illustration"]):
            analysis["style"] = "cartoon"
        elif any(word in prompt_lower for word in ["abstract", "modern", "minimalist"]):
            analysis["style"] = "abstract"
        elif any(word in prompt_lower for word in ["vintage", "retro", "classic"]):
            analysis["style"] = "vintage"
        
        # Detect mood keywords
        if any(word in prompt_lower for word in ["dark", "mysterious", "gothic"]):
            analysis["mood"] = "dark"
        elif any(word in prompt_lower for word in ["bright", "cheerful", "happy"]):
            analysis["mood"] = "bright"
        elif any(word in prompt_lower for word in ["dramatic", "epic", "heroic"]):
            analysis["mood"] = "dramatic"
        
        # Detect lighting keywords
        if any(word in prompt_lower for word in ["sunset", "golden hour", "warm"]):
            analysis["lighting"] = "warm"
        elif any(word in prompt_lower for word in ["night", "moonlight", "blue"]):
            analysis["lighting"] = "cool"
        elif any(word in prompt_lower for word in ["dramatic", "contrast", "shadow"]):
            analysis["lighting"] = "dramatic"
        
        # Add suggestions based on analysis
        if analysis["word_count"] < 5:
            analysis["suggestions"].append("Consider adding more descriptive details")
        if analysis["style"] == "realistic":
            analysis["suggestions"].append("Consider specifying an artistic style")
        if "lighting" not in prompt_lower:
            analysis["suggestions"].append("Consider adding lighting details")
        
        return analysis

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a prompt with additional details and context.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Enhanced prompt with additional details
        """
        if self.model and self.tokenizer:
            return self._enhance_with_local_model(prompt)
        else:
            return self._enhance_with_template(prompt)

    def _enhance_with_local_model(self, prompt: str) -> str:
        """Enhance prompt using local model."""
        try:
            if not self.model or not self.tokenizer:
                return self._enhance_with_template(prompt)
                
            system_prompt = """Enhance the following creative prompt by adding:
1. Descriptive adjectives
2. Lighting and atmosphere details
3. Artistic style suggestions
4. Quality indicators

Make it more detailed and creative while keeping the original intent.
Return only the enhanced prompt, no explanations."""

            full_prompt = f"{system_prompt}\n\nOriginal: {prompt}\n\nEnhanced:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if full_prompt in response:
                enhanced = response.split(full_prompt)[-1].strip()
                return enhanced if enhanced else self._enhance_with_template(prompt)
            else:
                return self._enhance_with_template(prompt)
            
        except Exception as e:
            logger.error(f"Local model enhancement failed: {e}")
            return self._enhance_with_template(prompt)

    def _enhance_with_template(self, prompt: str) -> str:
        """Enhance prompt using template-based rules."""
        # Template-based enhancement
        styles = [
            "photorealistic", "detailed", "high-quality", "professional",
            "cinematic", "artistic", "stunning", "beautiful"
        ]
        
        lighting = [
            "perfect lighting", "dramatic shadows", "cinematic lighting",
            "studio lighting", "natural lighting", "golden hour"
        ]
        
        details = [
            "detailed textures", "crisp details", "rich colors",
            "high resolution", "professional composition", "gallery-worthy"
        ]
        
        techniques = [
            "masterful technique", "artistic style", "professional quality",
            "digital art style", "photography style"
        ]
        
        # Select random enhancements
        style = random.choice(styles)
        light = random.choice(lighting)
        detail = random.choice(details)
        technique = random.choice(techniques)
        
        enhanced = f"{style} {prompt} with {light}, {detail}, and {technique}"
        return enhanced

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt for potential issues.
        
        Args:
            prompt: The prompt to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check for inappropriate content
        inappropriate_words = ["nude", "naked", "explicit", "adult", "nsfw"]
        if any(word in prompt.lower() for word in inappropriate_words):
            issues.append("Contains potentially inappropriate content")
        
        # Check for very short prompts
        if len(prompt.split()) < 2:
            warnings.append("Very short prompt - consider adding more details")
        
        # Check for very long prompts
        if len(prompt.split()) > 50:
            warnings.append("Very long prompt - consider simplifying")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

    def generate_variations(self, prompt: str, count: int = 3) -> List[str]:
        """
        Generate variations of a prompt.
        
        Args:
            prompt: The original prompt
            count: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        if self.model and self.tokenizer:
            return self._generate_variations_with_local_model(prompt, count)
        else:
            return self._generate_variations_with_template(prompt, count)

    def _generate_variations_with_local_model(self, prompt: str, count: int) -> List[str]:
        """Generate variations using local model."""
        try:
            if not self.model or not self.tokenizer:
                return self._generate_variations_with_template(prompt, count)
                
            system_prompt = f"""Generate {count} creative variations of the following prompt.
Each variation should be different but maintain the same core concept.
Return only the variations, one per line, no numbering."""

            full_prompt = f"{system_prompt}\n\nOriginal: {prompt}\n\nVariations:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if full_prompt in response:
                variations_text = response.split(full_prompt)[-1].strip()
                variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
                return variations[:count]
            else:
                return self._generate_variations_with_template(prompt, count)
            
        except Exception as e:
            logger.error(f"Local model variation generation failed: {e}")
            return self._generate_variations_with_template(prompt, count)

    def _generate_variations_with_template(self, prompt: str, count: int) -> List[str]:
        """Generate variations using template-based rules."""
        variations = []
        
        # Different styles
        styles = ["photorealistic", "artistic", "cinematic", "fantasy", "sci-fi"]
        
        # Different lighting
        lighting = ["sunset", "moonlight", "studio lighting", "natural light", "dramatic shadows"]
        
        # Different perspectives
        perspectives = ["close-up", "wide shot", "aerial view", "portrait", "landscape"]
        
        for i in range(count):
            if i == 0:
                # First variation: different style
                style = random.choice(styles)
                variation = f"{style} {prompt}"
            elif i == 1:
                # Second variation: different lighting
                light = random.choice(lighting)
                variation = f"{prompt} in {light}"
            else:
                # Third variation: different perspective
                perspective = random.choice(perspectives)
                variation = f"{perspective} of {prompt}"
            
            variations.append(variation)
        
        return variations[:count]

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Chat with the LLM.
        
        Args:
            message: User message
            history: Conversation history
            
        Returns:
            LLM response
        """
        if self.model and self.tokenizer:
            return self._chat_with_local_model(message, history)
        elif self.use_hf_api:
            return self._chat_with_hf_api(message, history)
        else:
            return self._chat_with_template(message, history)

    def _chat_with_local_model(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Chat using local model."""
        try:
            if not self.model or not self.tokenizer:
                return self._chat_with_template(message, history)
                
            # Build conversation context
            conversation = ""
            if history:
                for turn in history:
                    conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            
            full_prompt = f"{conversation}User: {message}\nAssistant:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if full_prompt in response:
                return response.split(full_prompt)[-1].strip()
            else:
                return response.strip()
            
        except Exception as e:
            logger.error(f"Local model chat failed: {e}")
            return self._chat_with_template(message, history)

    def _chat_with_hf_api(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Chat using Hugging Face API."""
        try:
            api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not api_token:
                return self._chat_with_template(message, history)
            
            model_id = "microsoft/DialoGPT-medium"
            headers = {"Authorization": f"Bearer {api_token}"}
            
            # Build conversation
            conversation = ""
            if history:
                for turn in history:
                    conversation += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            
            full_prompt = f"{conversation}User: {message}\nAssistant:"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    response_text = result[0].get("generated_text", "")
                    if full_prompt in response_text:
                        return response_text.split(full_prompt)[-1].strip()
                    return response_text.strip()
            
            return self._chat_with_template(message, history)
            
        except Exception as e:
            logger.error(f"Hugging Face API chat failed: {e}")
            return self._chat_with_template(message, history)

    def _chat_with_template(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Chat using template-based approach."""
        responses = [
            "I understand you're asking about creative content. Let me help you with that.",
            "That's an interesting creative request. I can assist you with generating content based on your input.",
            "I'm here to help with your creative projects. What specific details would you like me to focus on?",
            "Great question! For creative content generation, I can help expand your ideas and provide detailed descriptions.",
            "I'm ready to help with your creative needs. Let's work together to bring your vision to life."
        ]
        
        import random
        return random.choice(responses) 