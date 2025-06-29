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

# Try to import llama-cpp-python for local LLM support
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Using template-based processing only.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterface:
    """
    LLM Interface provides intelligent prompt enhancement and analysis.
    Uses template-based expansion for consistent, high-quality results.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path to local LLM model (.gguf file)
        """
        self.llm = None
        self.model_path = model_path or self._find_model()
        
        if self.model_path and LLAMA_AVAILABLE:
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                logger.info(f"✅ Local LLM loaded: {self.model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load local LLM: {e}")
                self.llm = None
        
        if not self.llm:
            logger.info("📝 Using template-based prompt processing")

    def _find_model(self) -> Optional[str]:
        """Find a local LLM model file."""
        possible_paths = [
            "llama-2-7b-chat.gguf",
            "../llama-2-7b-chat.gguf",
            "models/llama-2-7b-chat.gguf",
            "*.gguf"  # Any GGUF file
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Search for any .gguf file
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".gguf"):
                    return os.path.join(root, file)
        
        return None

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt for quality, complexity, and suggestions.
        
        Args:
            prompt: The input prompt to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if self.llm:
            return self._analyze_with_llm(prompt)
        else:
            return self._analyze_with_templates(prompt)

    def _analyze_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using local LLM."""
        try:
            system_prompt = """Analyze the following creative prompt and provide:
1. Word count
2. Estimated quality (low/medium/high)
3. Complexity level (low/medium/high)
4. Suggestions for improvement
5. Potential issues

Format your response as JSON:
{
    "word_count": number,
    "estimated_quality": "low|medium|high",
    "complexity": "low|medium|high",
    "suggestions": ["suggestion1", "suggestion2"],
    "potential_issues": ["issue1", "issue2"]
}"""

            response = self.llm(
                f"{system_prompt}\n\nPrompt: {prompt}\n\nAnalysis:",
                max_tokens=500,
                temperature=0.1,
                stop=["\n\n"]
            )
            
            # Try to parse JSON from response
            import json
            try:
                analysis = json.loads(response['choices'][0]['text'].strip())
                return {
                    "original_prompt": prompt,
                    **analysis
                }
            except:
                # Fallback to template analysis
                return self._analyze_with_templates(prompt)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._analyze_with_templates(prompt)

    def _analyze_with_templates(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using template-based rules."""
        words = prompt.split()
        word_count = len(words)
        
        # Quality estimation based on word count and content
        if word_count < 3:
            quality = "low"
        elif word_count < 8:
            quality = "medium"
        else:
            quality = "high"
        
        # Complexity estimation
        complex_words = ["detailed", "intricate", "complex", "sophisticated", "advanced"]
        complexity = "high" if any(word in prompt.lower() for word in complex_words) else "medium"
        
        # Suggestions
        suggestions = []
        if word_count < 5:
            suggestions.append("Consider adding more descriptive details")
        if not any(adj in prompt.lower() for adj in ["beautiful", "stunning", "amazing", "detailed"]):
            suggestions.append("Consider adding descriptive adjectives")
        if word_count < 8:
            suggestions.append("Use prompt enhancement for better results")
        
        return {
            "original_prompt": prompt,
            "word_count": word_count,
            "estimated_quality": quality,
            "complexity": complexity,
            "suggestions": suggestions,
            "potential_issues": []
        }

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a prompt with additional details and context.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Enhanced prompt with additional details
        """
        if self.llm:
            return self._enhance_with_llm(prompt)
        else:
            return self._enhance_with_templates(prompt)

    def _enhance_with_llm(self, prompt: str) -> str:
        """Enhance prompt using local LLM."""
        try:
            system_prompt = """Enhance the following creative prompt by adding:
1. Descriptive adjectives
2. Lighting and atmosphere details
3. Artistic style suggestions
4. Quality indicators

Make it more detailed and creative while keeping the original intent.
Return only the enhanced prompt, no explanations."""

            response = self.llm(
                f"{system_prompt}\n\nOriginal: {prompt}\n\nEnhanced:",
                max_tokens=200,
                temperature=0.7,
                stop=["\n\n"]
            )
            
            enhanced = response['choices'][0]['text'].strip()
            return enhanced if enhanced else self._enhance_with_templates(prompt)
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return self._enhance_with_templates(prompt)

    def _enhance_with_templates(self, prompt: str) -> str:
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
        if self.llm:
            return self._generate_variations_with_llm(prompt, count)
        else:
            return self._generate_variations_with_templates(prompt, count)

    def _generate_variations_with_llm(self, prompt: str, count: int) -> List[str]:
        """Generate variations using local LLM."""
        try:
            system_prompt = f"""Generate {count} creative variations of the following prompt.
Each variation should be different but maintain the same core concept.
Return only the variations, one per line, no numbering."""

            response = self.llm(
                f"{system_prompt}\n\nOriginal: {prompt}\n\nVariations:",
                max_tokens=300,
                temperature=0.8,
                stop=["\n\n"]
            )
            
            variations = response['choices'][0]['text'].strip().split('\n')
            return [v.strip() for v in variations if v.strip()][:count]
            
        except Exception as e:
            logger.error(f"LLM variation generation failed: {e}")
            return self._generate_variations_with_templates(prompt, count)

    def _generate_variations_with_templates(self, prompt: str, count: int) -> List[str]:
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