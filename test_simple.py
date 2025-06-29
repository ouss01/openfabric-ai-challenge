#!/usr/bin/env python3
"""
🧪 Simple Test Script for AI Creative Pipeline
Tests core components without requiring Openfabric connections
"""

import sys
import os

def test_imports():
    """Test that core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from core.llm_interface import LLMInterface
        print("✅ LLMInterface imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LLMInterface: {e}")
        return False
    
    try:
        from core.memory import MemorySystem
        print("✅ MemorySystem imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MemorySystem: {e}")
        return False
    
    try:
        from server import MockCreativePipeline
        print("✅ MockCreativePipeline imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MockCreativePipeline: {e}")
        return False
    
    return True

def test_llm_interface():
    """Test the LLM interface."""
    print("\n🧠 Testing LLM Interface...")
    
    try:
        from core.llm_interface import LLMInterface
        
        llm = LLMInterface()
        
        # Test prompt enhancement
        test_prompt = "a dragon in a forest"
        enhanced = llm.enhance_prompt(test_prompt)
        
        print(f"✅ Prompt enhancement: '{test_prompt}' -> '{enhanced}'")
        
        # Test prompt analysis
        analysis = llm.analyze_prompt(test_prompt)
        print(f"✅ Prompt analysis: {analysis['estimated_quality']} quality")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Interface test failed: {e}")
        return False

def test_memory_system():
    """Test the memory system."""
    print("\n💾 Testing Memory System...")
    
    try:
        from core.memory import MemorySystem
        
        # Use a test database
        memory = MemorySystem("test_memory.db")
        
        # Test saving a creation
        memory_id = memory.save_creation(
            original_prompt="test prompt",
            enhanced_prompt="enhanced test prompt",
            image_path="test_image.png",
            model_3d_path="test_model.glb"
        )
        
        print(f"✅ Creation saved with ID: {memory_id}")
        
        # Test retrieving the creation
        creation = memory.get_creation(memory_id)
        if creation:
            print(f"✅ Creation retrieved: {creation['original_prompt']}")
        else:
            print("❌ Failed to retrieve creation")
            return False
        
        # Test search
        results = memory.search_creations("test", limit=5)
        print(f"✅ Search returned {len(results)} results")
        
        # Test stats
        stats = memory.get_creation_stats()
        print(f"✅ Memory stats: {stats['total_creations']} total creations")
        
        # Clean up test database - try multiple times with delay
        import time
        for attempt in range(3):
            try:
                if os.path.exists("test_memory.db"):
                    os.remove("test_memory.db")
                    print("✅ Test database cleaned up")
                    break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)  # Wait 500ms before retry
                    continue
                else:
                    print("⚠️  Could not delete test database (file may be in use)")
                    # Don't fail the test for this
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ Memory System test failed: {e}")
        return False

def test_mock_pipeline():
    """Test the mock pipeline."""
    print("\n🎨 Testing Mock Pipeline...")
    
    try:
        from server import MockCreativePipeline
        
        pipeline = MockCreativePipeline()
        
        # Test prompt analysis
        analysis = pipeline.analyze_prompt("a beautiful sunset")
        print(f"✅ Pipeline prompt analysis: {analysis['estimated_quality']} quality")
        
        # Test mock creation
        result = pipeline.create_from_prompt("a majestic dragon")
        if result["success"]:
            print(f"✅ Mock pipeline created: {result['memory_id']}")
            print(f"   Image: {result['image_path']}")
            print(f"   Model: {result['model_3d_path']}")
        else:
            print(f"❌ Mock pipeline failed: {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Mock Pipeline test failed: {e}")
        return False

def test_directories():
    """Test that required directories exist."""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = ['outputs', 'outputs/images', 'outputs/models']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"⚠️  Directory missing: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 AI Creative Pipeline - Simple Component Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("LLM Interface", test_llm_interface),
        ("Memory System", test_memory_system),
        ("Mock Pipeline", test_mock_pipeline),
        ("Directory Structure", test_directories)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AI Creative Pipeline is ready to use.")
        print("\n🎯 Next steps:")
        print("1. Run the server: poetry run uvicorn server:app --host 0.0.0.0 --port 8888 --reload")
        print("2. Access Swagger UI: http://localhost:8888/docs")
        print("3. Configure with mock_mode: true for testing")
        print("4. Start creating amazing content!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 