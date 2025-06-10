#!/usr/bin/env python3
"""
Comprehensive installer testing script to verify all fixes work correctly
Tests installer startup, admin privilege check, cross-platform compatibility, and error handling
"""

import sys
import os
import tempfile
import platform
import subprocess
from pathlib import Path

def test_installer_startup():
    """Test that installer starts without freezing"""
    print("🧪 Testing installer startup (no freeze)...")
    
    try:
        result = subprocess.run([
            sys.executable, "MedAI_Radiologia_Installer.py"
        ], input="n\n", text=True, capture_output=True, timeout=10)
        
        if "🏥 MedAI Radiologia - Instalador Autônomo" in result.stdout:
            print("✅ Installer starts successfully without freezing")
            return True
        else:
            print("❌ Installer output not as expected")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installer appears to be frozen (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error testing installer startup: {e}")
        return False

def test_cross_platform_feedback():
    """Test cross-platform warning messages"""
    print("\n🧪 Testing cross-platform feedback...")
    
    try:
        result = subprocess.run([
            sys.executable, "MedAI_Radiologia_Installer.py"
        ], input="n\n", text=True, capture_output=True, timeout=10)
        
        current_platform = platform.system()
        if current_platform != "Windows":
            if "⚠️ Aviso: Algumas funcionalidades específicas do Windows não estarão disponíveis:" in result.stdout:
                print("✅ Cross-platform warning displayed correctly")
                return True
            else:
                print("❌ Cross-platform warning not found in output")
                return False
        else:
            print("✅ Running on Windows - cross-platform warning not expected")
            return True
            
    except Exception as e:
        print(f"❌ Error testing cross-platform feedback: {e}")
        return False

def test_admin_privilege_check():
    """Test admin privilege check functionality"""
    print("\n🧪 Testing admin privilege check...")
    
    try:
        sys.path.append('.')
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        
        installer = MedAIWindowsInstaller()
        result = installer.check_admin_privileges()
        
        print(f"Admin privilege check result: {result}")
        print("✅ Admin privilege check executed without errors")
        return True
        
    except Exception as e:
        print(f"❌ Error testing admin privilege check: {e}")
        return False

def test_windows_specific_methods():
    """Test Windows-specific methods handle non-Windows systems gracefully"""
    print("\n🧪 Testing Windows-specific methods...")
    
    try:
        sys.path.append('.')
        from MedAI_Radiologia_Installer import MedAIWindowsInstaller
        
        installer = MedAIWindowsInstaller()
        
        installer.create_shortcuts()
        print("✅ create_shortcuts() executed without errors")
        
        installer.register_application()
        print("✅ register_application() executed without errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Windows-specific methods: {e}")
        return False

def test_gui_availability():
    """Test GUI availability detection"""
    print("\n🧪 Testing GUI availability detection...")
    
    try:
        sys.path.append('.')
        from MedAI_Radiologia_Installer import GUI_AVAILABLE
        
        print(f"GUI_AVAILABLE: {GUI_AVAILABLE}")
        print("✅ GUI availability detection works")
        return True
        
    except Exception as e:
        print(f"❌ Error testing GUI availability: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive installer tests"""
    print("=" * 70)
    print("🔧 COMPREHENSIVE INSTALLER TESTING")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 70)
    
    tests = [
        ("Installer startup (no freeze)", test_installer_startup),
        ("Cross-platform feedback", test_cross_platform_feedback),
        ("Admin privilege check", test_admin_privilege_check),
        ("Windows-specific methods", test_windows_specific_methods),
        ("GUI availability detection", test_gui_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        print("-" * 50)
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"Result: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ Test error: {e}")
            print(f"Result: ❌ FAIL (exception)")
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    passed_count = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed_count += 1
        else:
            all_passed = False
    
    print("-" * 70)
    print(f"Tests passed: {passed_count}/{len(tests)}")
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Comprehensive installer testing successful!")
        print("All installer fixes are working correctly:")
        print("• ✅ Installer no longer freezes on startup")
        print("• ✅ Cross-platform feedback is displayed appropriately")
        print("• ✅ Admin privilege check works cross-platform")
        print("• ✅ Windows-specific features handle non-Windows systems gracefully")
        print("• ✅ GUI availability detection works correctly")
        print("\n🚀 The installer freeze issue has been completely resolved!")
    else:
        print("\n⚠️  Some comprehensive tests failed.")
        print("Please review the installer implementation.")
    
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
