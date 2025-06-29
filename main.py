import logging
import os
from typing import Dict, Optional
from datetime import datetime

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.pipeline import CreativePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Global pipeline instance
pipeline: Optional[CreativePipeline] = None

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data and initializes the creative pipeline.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application.
    """
    global pipeline

    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

        # Try to find local LLaMA .gguf model
        model_path = None
        for file in os.listdir(os.getcwd()):
            if file.endswith(".gguf"):
                model_path = os.path.join(os.getcwd(), file)
                logging.info(f"🧠 Found local LLaMA model: {model_path}")
                break

        # Initialize pipeline with app IDs and LLaMA model path
        if conf.app_ids and len(conf.app_ids) >= 2:
            try:
                pipeline = CreativePipeline(app_ids=conf.app_ids, local_llm_path=model_path)
                logging.info(f"Creative Pipeline initialized with {len(conf.app_ids)} app IDs and LLaMA: {bool(model_path)}")
            except Exception as e:
                logging.error(f"Failed to initialize Creative Pipeline: {e}")
                pipeline = None
        else:
            logging.warning("Insufficient app IDs provided. Need at least 2 for text-to-image and image-to-3d")


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    global pipeline

    try:
        request: InputClass = model.request
        response: OutputClass = model.response

        if not request.prompt or not request.prompt.strip():
            response.message = "❌ Error: No prompt provided. Please provide a creative prompt."
            logging.error("No prompt provided in request")
            return

        prompt = request.prompt.strip()
        logging.info(f"🎨 Processing creative request: {prompt}")

        if not pipeline:
            response.message = "❌ Error: Creative Pipeline not initialized. Please configure app IDs first."
            logging.error("Pipeline not initialized - app IDs not configured")
            return

        result = pipeline.create_from_prompt(prompt)

        if result["success"]:
            response.message = result["message"]
            logging.info(f"✅ Creative pipeline completed successfully in {result['processing_time']:.2f}s")
            logging.info(f"📊 Pipeline Stats: Memory ID={result['memory_id']}, "
                         f"Image={result['image_path']}, 3D Model={result['model_3d_path']}")
        else:
            response.message = result["message"]
            logging.error(f"❌ Creative pipeline failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logging.error(f"Unexpected error in execute function: {e}")
        model.response.message = f"❌ Unexpected error: {str(e)}"


############################################################
# Utility functions for additional features
############################################################
def search_creations(query: str, limit: int = 5) -> Dict:
    global pipeline
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    try:
        results = pipeline.search_memories(query, limit)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return {"error": str(e)}


def get_recent_creations(limit: int = 10) -> Dict:
    global pipeline
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    try:
        results = pipeline.get_recent_creations(limit)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        logging.error(f"Failed to get recent creations: {e}")
        return {"error": str(e)}


def get_pipeline_stats() -> Dict:
    global pipeline
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    try:
        stats = pipeline.get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logging.error(f"Failed to get pipeline stats: {e}")
        return {"error": str(e)}


def analyze_prompt(prompt: str) -> Dict:
    global pipeline
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    try:
        analysis = pipeline.analyze_prompt(prompt)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        logging.error(f"Failed to analyze prompt: {e}")
        return {"error": str(e)}


# Initial logs
logging.info("🚀 AI Creative Pipeline - Application Starting")
logging.info("=" * 60)
logging.info("🎯 Mission: Transform simple text prompts into stunning images and 3D models")
logging.info("🧠 Features: Local LLM processing, Openfabric integration, persistent memory")
logging.info("📁 Output: Images and 3D models with complete metadata tracking")
logging.info("=" * 60)
