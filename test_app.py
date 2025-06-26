#!/usr/bin/env python3
"""
Test script untuk memverifikasi aplikasi Streamlit WDBC
"""

import sys
import importlib
import os

def test_imports():
    """Test semua import yang diperlukan"""
    print("🧪 Testing imports...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'matplotlib',
        'seaborn',
        'sklearn',
        'pickle'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_model_files():
    """Test keberadaan file model"""
    print("\n🧪 Testing model files...")
    
    required_files = [
        'models/best_model_support_vector_machine.pkl',
        'models/scaler.pkl',
        'models/label_encoder.pkl',
        'models/model_info.json'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All model files found!")
        return True

def test_data_file():
    """Test keberadaan file data"""
    print("\n🧪 Testing data file...")
    
    data_file = 'wdbc_clean_no_outliers.csv'
    
    if os.path.exists(data_file):
        print(f"✅ {data_file}")
        return True
    else:
        print(f"⚠️ {data_file} (optional)")
        return True

def test_streamlit_app():
    """Test aplikasi Streamlit"""
    print("\n🧪 Testing Streamlit app...")
    
    try:
        # Import aplikasi
        import streamlit_app
        
        # Test fungsi load_model
        model, scaler, label_encoder, model_info = streamlit_app.load_model()
        
        if model is not None and scaler is not None and label_encoder is not None:
            print("✅ Model loaded successfully")
            print(f"✅ Model type: {model_info.get('model_type', 'Unknown')}")
            return True
        else:
            print("❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ Error testing app: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 WDBC Breast Cancer Prediction - App Test")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Data File", test_data_file),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! App is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 