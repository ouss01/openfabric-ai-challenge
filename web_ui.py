import streamlit as st
import requests
import json
import time
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="AI Creative Pipeline",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8888"

# Add this after API_BASE_URL and before any function definitions
DEFAULT_APP_IDS = [
    "f0997a01-d6d3-a5fe-53d8-561300318557",
    "69543f29-4d41-4afc-7f29-3d51591f11eb"
]
DEFAULT_MOCK_MODE = False

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def configure_pipeline(app_ids, mock_mode):
    """Configure the pipeline with app IDs and mode."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/config",
            json={"app_ids": app_ids, "mock_mode": mock_mode},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_content(prompt):
    """Generate content using the pipeline."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/execute",
            json={"prompt": prompt},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_creations(query, limit=10):
    """Search through past creations."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "limit": limit},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_recent_creations(limit=10):
    """Get recent creations."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/recent?limit={limit}",
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_prompt(prompt):
    """Analyze a prompt without generating content."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"prompt": prompt},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Creative Pipeline</h1>', unsafe_allow_html=True)
    
    # Auto-configure pipeline if not already configured
    if 'pipeline_configured' not in st.session_state:
        with st.spinner("Auto-configuring pipeline..."):
            result = configure_pipeline(DEFAULT_APP_IDS, DEFAULT_MOCK_MODE)
            if result.get("success"):
                st.session_state.pipeline_configured = True
                st.success("✅ Pipeline auto-configured!")
            else:
                st.session_state.pipeline_configured = False
                st.error(f"❌ Auto-configuration failed: {result.get('error', 'Unknown error')}")
    
    # Check API health
    if not check_api_health():
        st.error("❌ API server is not running. Please start the server first.")
        st.info("Run: `poetry run uvicorn server:app --host 0.0.0.0 --port 8888 --reload`")
        return
    
    st.success("✅ API server is running!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pipeline configuration
        st.subheader("Pipeline Settings")
        
        app_ids_input = st.text_area(
            "Openfabric App IDs (one per line)",
            value="f0997a01-d6d3-a5fe-53d8-561300318557\n69543f29-4d41-4afc-7f29-3d51591f11eb",
            help="Enter Openfabric app IDs for enhanced generation"
        )
        
        app_ids = [aid.strip() for aid in app_ids_input.split('\n') if aid.strip()]
        
        mock_mode = st.checkbox(
            "Mock Mode",
            value=False,
            help="Use mock mode for testing without external APIs"
        )
        
        if st.button("🔧 Configure Pipeline"):
            with st.spinner("Configuring pipeline..."):
                result = configure_pipeline(app_ids, mock_mode)
                if result.get("success"):
                    st.success("✅ Pipeline configured successfully!")
                    st.json(result)
                else:
                    st.error(f"❌ Configuration failed: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("📊 View Recent Creations"):
            st.session_state.show_recent = True
        
        if st.button("🔍 Search Creations"):
            st.session_state.show_search = True
        
        if st.button("📈 View Statistics"):
            st.session_state.show_stats = True
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Generate", "🔍 Search", "📊 Analytics", "📁 Recent"])
    
    with tab1:
        st.header("🎨 Generate Content")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your creative prompt",
            placeholder="e.g., a majestic dragon flying over a medieval castle at sunset",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Generate Content", type="primary"):
                if prompt.strip():
                    with st.spinner("🎨 Creating your masterpiece..."):
                        result = generate_content(prompt)
                        
                        if result.get("success"):
                            st.success("✅ Content generated successfully!")
                            
                            # Display enhanced prompt
                            enhanced_prompt = result.get("enhanced_prompt", "")
                            if enhanced_prompt:
                                st.markdown(f"**Enhanced Prompt:** {enhanced_prompt}")
                            else:
                                st.info("No enhanced prompt returned.")
                            
                            # Display generated image
                            image_path = result.get("image_path", "")
                            if image_path and os.path.exists(image_path):
                                st.image(image_path, caption="Generated Image")
                            elif image_path:
                                st.warning(f"Image file not found: {image_path}")
                            else:
                                st.info("No image generated.")
                            
                            # Display 3D model info
                            model_3d_path = result.get("model_3d_path", "")
                            if model_3d_path:
                                st.info(f"3D Model: {model_3d_path}")
                            
                            # Display metadata
                            st.subheader("📋 Generation Details")
                            col_c, col_d = st.columns(2)
                            with col_c:
                                st.write("**Original Prompt:**", result.get("original_prompt", ""))
                                st.write("**Memory ID:**", result.get("memory_id", ""))
                            with col_d:
                                st.write("**Processing Time:**", f"{result.get('processing_time', 0):.2f}s")
                                st.write("**Creation ID:**", result.get("creation_id", ""))
                            
                            # Display analysis
                            if "analysis" in result:
                                st.subheader("🔍 Prompt Analysis")
                                analysis = result["analysis"]
                                col_e, col_f, col_g = st.columns(3)
                                with col_e:
                                    st.metric("Word Count", analysis.get("word_count", 0))
                                with col_f:
                                    st.metric("Quality", analysis.get("estimated_quality", "Unknown"))
                                with col_g:
                                    st.metric("Complexity", analysis.get("complexity", "Unknown"))
                                if analysis.get("suggestions"):
                                    st.info("💡 Suggestions: " + ", ".join(analysis["suggestions"]))
                        else:
                            st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please enter a prompt")
        
        with col2:
            if st.button("🔍 Analyze Prompt"):
                if prompt.strip():
                    with st.spinner("Analyzing prompt..."):
                        result = analyze_prompt(prompt)
                        
                        if result.get("success"):
                            st.success("✅ Analysis complete!")
                            st.json(result)
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please enter a prompt")
            
            # New: Enhance Prompt button
            if st.button("✨ Enhance Prompt"):
                if prompt.strip():
                    with st.spinner("Enhancing prompt..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/enhance",
                                json={"prompt": prompt},
                                timeout=10
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("success"):
                                    st.success("✨ Enhanced Prompt:")
                                    st.info(data.get("enhanced_prompt", "No enhanced prompt returned."))
                                else:
                                    st.error(f"❌ Enhancement failed: {data.get('error', 'Unknown error')}")
                            else:
                                st.error(f"❌ Enhancement failed: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Enhancement error: {e}")
                else:
                    st.warning("⚠️ Please enter a prompt")
    
    with tab2:
        st.header("🔍 Search Creations")
        
        search_query = st.text_input("Search query", placeholder="e.g., dragon, castle, sunset")
        search_limit = st.slider("Results limit", 1, 50, 10)
        
        if st.button("🔍 Search"):
            if search_query.strip():
                with st.spinner("Searching..."):
                    results = search_creations(search_query, search_limit)
                    
                    if results.get("success"):
                        st.success(f"✅ Found {len(results.get('results', []))} results")
                        
                        for i, creation in enumerate(results.get("results", [])):
                            with st.expander(f"Creation {i+1}: {creation.get('original_prompt', 'Unknown')}"):
                                st.write("**Original Prompt:**", creation.get("original_prompt", ""))
                                st.write("**Enhanced Prompt:**", creation.get("enhanced_prompt", ""))
                                st.write("**Memory ID:**", creation.get("memory_id", ""))
                                st.write("**Timestamp:**", creation.get("timestamp", ""))
                    else:
                        st.error(f"❌ Search failed: {results.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Please enter a search query")
    
    with tab3:
        st.header("📊 Analytics")
        
        if st.button("📈 Get Statistics"):
            with st.spinner("Loading statistics..."):
                try:
                    response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
                    if response.status_code == 200:
                        stats = response.json()
                        st.success("✅ Statistics loaded!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Creations", stats.get("total_creations", 0))
                        
                        with col2:
                            st.metric("Average Processing Time", f"{stats.get('avg_processing_time', 0):.2f}s")
                        
                        with col3:
                            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
                        
                        st.json(stats)
                    else:
                        st.error("❌ Failed to load statistics")
                except Exception as e:
                    st.error(f"❌ Error loading statistics: {e}")
    
    with tab4:
        st.header("📁 Recent Creations")
        
        recent_limit = st.slider("Number of recent creations", 1, 20, 5)
        
        if st.button("🔄 Refresh Recent"):
            with st.spinner("Loading recent creations..."):
                results = get_recent_creations(recent_limit)
                
                if results.get("success"):
                    st.success(f"✅ Loaded {len(results.get('results', []))} recent creations")
                    
                    for i, creation in enumerate(results.get("results", [])):
                        with st.expander(f"Recent {i+1}: {creation.get('original_prompt', 'Unknown')}"):
                            st.write("**Original Prompt:**", creation.get("original_prompt", ""))
                            st.write("**Enhanced Prompt:**", creation.get("enhanced_prompt", ""))
                            st.write("**Memory ID:**", creation.get("memory_id", ""))
                            st.write("**Timestamp:**", creation.get("timestamp", ""))
                            
                            if creation.get("image_path") and os.path.exists(creation["image_path"]):
                                st.image(creation["image_path"], caption="Generated Image", width=200)
                else:
                    st.error(f"❌ Failed to load recent creations: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 