import logging
import requests
import json
import time
from typing import Dict, Any, Optional, List

# Try to import Openfabric SDK
try:
    from openfabric_pysdk import Proxy
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenfabricIntegration:
    """
    Integration with Openfabric apps using SDK (Proxy) or HTTP APIs.
    """
    
    def __init__(self):
        """Initialize the Openfabric integration."""
        self.app_configs = {}
        self.sdk_proxies = {}  # app_id -> Proxy
        
    def configure_apps(self, app_ids: List[str]) -> Dict[str, Any]:
        """
        Configure Openfabric apps with the provided app IDs.
        
        Args:
            app_ids: List of Openfabric app IDs to configure
            
        Returns:
            Configuration status
        """
        try:
            results = {}
            
            for app_id in app_ids:
                logger.info(f"🔧 Configuring Openfabric app: {app_id}")
                
                sdk_status = None
                if SDK_AVAILABLE:
                    try:
                        proxy = Proxy(app_id)
                        self.sdk_proxies[app_id] = proxy
                        sdk_status = "sdk"
                        logger.info(f"✅ SDK Proxy created for {app_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ SDK Proxy failed for {app_id}: {e}")
                        sdk_status = None
                # Always try HTTP as well for manifest/info
                app_config = self._connect_to_app(app_id)
                if app_config:
                    app_config['sdk_status'] = sdk_status
                    self.app_configs[app_id] = app_config
                    results[app_id] = {
                        "status": "success",
                        "sdk": bool(sdk_status),
                        "name": app_config.get("name", "Unknown")
                    }
                    logger.info(f"✅ Successfully configured app: {app_id}")
                else:
                    results[app_id] = {
                        "status": "failed",
                        "sdk": False,
                        "error": "Could not connect to app"
                    }
                    logger.warning(f"❌ Failed to configure app: {app_id}")
            
            return {
                "success": any(r["status"] == "success" for r in results.values()),
                "apps": results,
                "total_configured": sum(1 for r in results.values() if r["status"] == "success")
            }
            
        except Exception as e:
            logger.error(f"❌ Error in app configuration: {e}")
            return {
                "success": False,
                "error": str(e),
                "apps": {}
            }
    
    def _connect_to_app(self, app_id: str) -> Optional[Dict[str, Any]]:
        """
        Connect to a specific Openfabric app.
        
        Args:
            app_id: The app ID to connect to
            
        Returns:
            App configuration if successful, None otherwise
        """
        try:
            # Try different connection methods
            connection_methods = [
                self._try_direct_connection,
                self._try_api_connection
            ]
            
            for method in connection_methods:
                try:
                    config = method(app_id)
                    if config:
                        return config
                except Exception as e:
                    logger.debug(f"Connection method failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error connecting to app {app_id}: {e}")
            return None
    
    def _try_direct_connection(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Try direct connection to the app."""
        try:
            # Try to get app manifest
            manifest_urls = [
                f"https://{app_id}/manifest",
                f"https://api.openfabric.network/app/{app_id}/manifest",
                f"https://openfabric.network/api/app/{app_id}/manifest"
            ]
            
            for url in manifest_urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        manifest = response.json()
                        return {
                            "id": app_id,
                            "name": manifest.get("name", "Unknown"),
                            "type": manifest.get("type", "Unknown"),
                            "manifest": manifest,
                            "connection_method": "direct"
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Direct connection failed: {e}")
            return None
    
    def _try_api_connection(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Try connection via Openfabric API."""
        try:
            # Try Openfabric network API
            api_urls = [
                f"https://api.openfabric.network/app/{app_id}",
                f"https://openfabric.network/api/app/{app_id}"
            ]
            
            for url in api_urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        app_data = response.json()
                        return {
                            "id": app_id,
                            "name": app_data.get("name", "Unknown"),
                            "type": app_data.get("type", "Unknown"),
                            "data": app_data,
                            "connection_method": "api"
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"API connection failed: {e}")
            return None
    
    def generate_image(self, app_id: str, prompt: str) -> Optional[bytes]:
        """
        Generate an image using an Openfabric app.
        
        Args:
            app_id: The app ID to use for generation
            prompt: The text prompt
            
        Returns:
            Image data if successful, None otherwise
        """
        try:
            if app_id not in self.app_configs:
                logger.warning(f"App {app_id} not configured")
                return None
            
            app_config = self.app_configs[app_id]
            connection_method = app_config.get("connection_method", "unknown")
            
            logger.info(f"🎨 Generating image with {app_id} using {connection_method}")
            
            if connection_method in ["direct", "api"]:
                return self._generate_with_http(app_config, prompt)
            else:
                logger.warning(f"Unknown connection method: {connection_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image with {app_id}: {e}")
            return None
    
    def _generate_with_http(self, app_config: Dict[str, Any], prompt: str) -> Optional[bytes]:
        """Generate image using HTTP API."""
        try:
            app_id = app_config["id"]
            
            # Try different API endpoints
            endpoints = [
                f"https://{app_id}/execute",
                f"https://{app_id}/generate",
                f"https://api.openfabric.network/app/{app_id}/execute",
                f"https://openfabric.network/api/app/{app_id}/execute"
            ]
            
            input_data = {
                "prompt": prompt,
                "parameters": {
                    "width": 512,
                    "height": 512,
                    "steps": 20,
                    "guidance_scale": 7.5
                }
            }
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=input_data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract image data
                        if "image" in result:
                            return result["image"]
                        elif "data" in result:
                            return result["data"]
                        elif "url" in result:
                            # Download image from URL
                            img_response = requests.get(result["url"])
                            if img_response.status_code == 200:
                                return img_response.content
                    
                except Exception as e:
                    logger.debug(f"HTTP endpoint {endpoint} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"HTTP generation failed: {e}")
            return None
    
    def generate_3d_model(self, app_id: str, prompt: str) -> Optional[bytes]:
        """
        Generate a 3D model using an Openfabric app.
        
        Args:
            app_id: The app ID to use for generation
            prompt: The text prompt
            
        Returns:
            3D model data if successful, None otherwise
        """
        try:
            if app_id not in self.app_configs:
                logger.warning(f"App {app_id} not configured")
                return None
            
            app_config = self.app_configs[app_id]
            connection_method = app_config.get("connection_method", "unknown")
            
            logger.info(f"🎲 Generating 3D model with {app_id} using {connection_method}")
            
            if connection_method in ["direct", "api"]:
                return self._generate_3d_with_http(app_config, prompt)
            else:
                logger.warning(f"Unknown connection method: {connection_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating 3D model with {app_id}: {e}")
            return None
    
    def _generate_3d_with_http(self, app_config: Dict[str, Any], prompt: str) -> Optional[bytes]:
        """Generate 3D model using HTTP API."""
        try:
            app_id = app_config["id"]
            
            # Try different API endpoints for 3D generation
            endpoints = [
                f"https://{app_id}/generate_3d",
                f"https://{app_id}/execute",
                f"https://api.openfabric.network/app/{app_id}/generate_3d",
                f"https://openfabric.network/api/app/{app_id}/generate_3d"
            ]
            
            input_data = {
                "prompt": prompt,
                "parameters": {
                    "format": "glb",
                    "resolution": "medium",
                    "style": "realistic"
                }
            }
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=input_data,
                        timeout=120  # Longer timeout for 3D generation
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract 3D model data
                        if "model" in result:
                            return result["model"]
                        elif "data" in result:
                            return result["data"]
                        elif "url" in result:
                            # Download model from URL
                            model_response = requests.get(result["url"])
                            if model_response.status_code == 200:
                                return model_response.content
                    
                except Exception as e:
                    logger.debug(f"HTTP 3D endpoint {endpoint} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"HTTP 3D generation failed: {e}")
            return None
    
    def get_app_status(self) -> Dict[str, Any]:
        """
        Get the status of configured apps.
        
        Returns:
            Status information for all configured apps
        """
        status = {
            "total_apps": len(self.app_configs),
            "configured_apps": [],
            "sdk_proxies": list(self.sdk_proxies.keys()),
            "connection_methods": {}
        }
        
        for app_id, config in self.app_configs.items():
            app_status = {
                "id": app_id,
                "name": config.get("name", "Unknown"),
                "type": config.get("type", "Unknown"),
                "connection_method": config.get("connection_method", "Unknown"),
                "sdk": app_id in self.sdk_proxies,
                "status": "configured"
            }
            status["configured_apps"].append(app_status)
            
            method = config.get("connection_method", "Unknown")
            status["connection_methods"][method] = status["connection_methods"].get(method, 0) + 1
        
        return status 