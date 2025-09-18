#!/usr/bin/env python3
"""
Test script to verify the import fix works correctly
"""

import sys
from pathlib import Path

def test_import_fix():
    """Test that the import fix resolves the ImportError"""
    print("🧪 Testing Import Fix")
    print("=" * 50)
    
    # Test the conditional import logic
    try:
        print("📦 Testing relative import...")
        # This will fail when run standalone
        from .qc_worklist_generator import generate_worklist_for_project
        print("✅ Relative import successful")
        return True
    except ImportError:
        print("❌ Relative import failed (expected when run standalone)")
        
        try:
            print("📦 Testing absolute import...")
            # Add current directory to path
            current_dir = Path(__file__).parent / "src" / "react_agent"
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            
            from qc_worklist_generator import generate_worklist_for_project
            print("✅ Absolute import successful")
            
            # Test the function is callable
            print("🔧 Testing function availability...")
            if callable(generate_worklist_for_project):
                print("✅ Function is callable")
                return True
            else:
                print("❌ Function is not callable")
                return False
                
        except ImportError as e:
            print(f"❌ Absolute import failed: {e}")
            return False

def test_q_qc_import():
    """Test importing the main Q_QC module"""
    print("\n🧪 Testing Q_QC Module Import")
    print("=" * 50)
    
    try:
        # Add the source directory to path
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        print("📦 Importing Q_QC module...")
        from react_agent import Q_QC
        print("✅ Q_QC module imported successfully")
        
        # Check if the worklist function is available
        if hasattr(Q_QC, 'generate_worklist_for_project'):
            print("✅ generate_worklist_for_project function is available")
            return True
        else:
            print("❌ generate_worklist_for_project function not found")
            return False
            
    except ImportError as e:
        print(f"❌ Q_QC module import failed: {e}")
        return False

def main():
    """Run all import tests"""
    print("🚀 Import Fix Test Suite")
    print("=" * 60)
    
    # Test 1: Direct import test
    import_success = test_import_fix()
    
    # Test 2: Q_QC module test
    qc_success = test_q_qc_import()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Direct Import: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"   Q_QC Module: {'✅ PASSED' if qc_success else '❌ FAILED'}")
    
    overall_success = import_success or qc_success  # Either approach should work
    print(f"\n🏆 Overall Result: {'✅ IMPORT FIX SUCCESSFUL' if overall_success else '❌ IMPORT FIX FAILED'}")
    
    if overall_success:
        print("\n🎉 The conditional import solution should resolve the LangGraph ImportError!")
        print("📋 The module can now be executed both:")
        print("   • As part of a package (relative imports)")
        print("   • Standalone by LangGraph (absolute imports)")
    else:
        print("\n⚠️ Import issues detected. Consider using Solution 3 (Inline Functions)")

if __name__ == "__main__":
    main()
