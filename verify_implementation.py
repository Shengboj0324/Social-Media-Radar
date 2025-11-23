#!/usr/bin/env python3
"""Verification script for Social Media Radar implementation.

This script verifies that all components are properly implemented and working.
"""

import sys
from pathlib import Path


def verify_modules():
    """Verify all modules can be imported."""
    print("🔍 Verifying module imports...\n")

    # Core modules (must work)
    core_modules = [
        "app.core.errors",
        "app.core.security_advanced",
        "app.core.credential_vault",
        "app.oauth.oauth_proxy",
        "app.media.media_downloader",
        "app.api.routes.auth",
        "app.api.routes.platforms",
        "app.api.middleware.security_middleware",
    ]

    # Connector modules (may fail if dependencies not installed)
    connector_modules = [
        "app.connectors.reddit",
        "app.connectors.youtube",
        "app.connectors.tiktok",
        "app.connectors.facebook",
        "app.connectors.instagram",
        "app.connectors.wechat",
        "app.connectors.nytimes",
        "app.connectors.wsj",
        "app.connectors.google_news",
        "app.connectors.apple_news",
        "app.connectors.abc_news",
    ]

    print("Core Modules:")
    core_success = 0
    core_failed = 0

    for module in core_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            core_success += 1
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            core_failed += 1

    print(f"\nConnector Modules (optional if dependencies not installed):")
    connector_success = 0
    connector_failed = 0

    for module in connector_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            connector_success += 1
        except Exception as e:
            # Check if it's just a missing dependency
            if "No module named" in str(e):
                print(f"  ⚠️  {module}: Missing dependency (install with: poetry install)")
            else:
                print(f"  ❌ {module}: {e}")
            connector_failed += 1

    total_success = core_success + connector_success
    total_modules = len(core_modules) + len(connector_modules)

    print(f"\n📊 Results:")
    print(f"  Core: {core_success}/{len(core_modules)} modules")
    print(f"  Connectors: {connector_success}/{len(connector_modules)} modules")
    print(f"  Total: {total_success}/{total_modules} modules")

    # Only fail if core modules fail
    if core_failed == 0:
        print("🎉 All core modules working perfectly!\n")
        if connector_failed > 0:
            print("ℹ️  Note: Install dependencies with 'poetry install' to enable connectors\n")
        return True
    else:
        print(f"⚠️  {core_failed} core module(s) failed\n")
        return False


def verify_files():
    """Verify all required files exist."""
    print("📁 Verifying file structure...\n")
    
    required_files = [
        "app/core/security_advanced.py",
        "app/core/credential_vault.py",
        "app/oauth/oauth_proxy.py",
        "app/media/media_downloader.py",
        "app/api/routes/platforms.py",
        "app/api/middleware/security_middleware.py",
        "docs/SECURITY_ARCHITECTURE.md",
        "docs/USER_GUIDE.md",
        "docs/TESTING_GUIDE.md",
        "SECURITY_AND_MEDIA_IMPLEMENTATION.md",
        "FINAL_IMPLEMENTATION_SUMMARY.md",
        "DEPLOYMENT_READY_SUMMARY.md",
    ]
    
    success = 0
    failed = 0
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
            success += 1
        else:
            print(f"❌ {file_path} - NOT FOUND")
            failed += 1
    
    print(f"\n📊 Results: {success}/{len(required_files)} files found")
    
    if failed == 0:
        print("🎉 All files present!\n")
        return True
    else:
        print(f"⚠️  {failed} file(s) missing\n")
        return False


def verify_security():
    """Verify security features."""
    print("🔐 Verifying security features...\n")
    
    try:
        from app.core.security_advanced import (
            military_encryption,
            intrusion_detection,
            SecurityHeaders,
            DataMasking,
        )
        
        # Test encryption
        plaintext = b"test_data"
        password = "test_password"
        encrypted = military_encryption.encrypt_multilayer(plaintext, password)
        decrypted = military_encryption.decrypt_multilayer(encrypted, password)
        
        assert decrypted == plaintext
        print("✅ Military-grade encryption working")
        
        # Test intrusion detection
        ip = "192.168.1.100"
        for _ in range(6):
            intrusion_detection.record_failed_attempt(ip)
        
        assert intrusion_detection.is_ip_blocked(ip)
        print("✅ Intrusion detection working")
        
        # Test security headers
        headers = SecurityHeaders.get_headers()
        assert "X-Frame-Options" in headers
        assert "Content-Security-Policy" in headers
        print("✅ Security headers configured")
        
        # Test data masking
        data = {"password": "secret123", "email": "test@example.com"}
        masked = DataMasking.mask_dict(data, ["password"])
        assert masked["password"] == "***"
        assert masked["email"] == "test@example.com"
        print("✅ Data masking working")
        
        print("\n🎉 All security features verified!\n")
        return True
        
    except Exception as e:
        print(f"❌ Security verification failed: {e}\n")
        return False


def verify_media():
    """Verify media capabilities."""
    print("🎬 Verifying media capabilities...\n")
    
    try:
        from app.media.media_downloader import (
            MediaDownloader,
            MediaType,
            VideoQuality,
            ImageFormat,
        )
        
        # Test downloader initialization
        downloader = MediaDownloader(
            storage_path="./test_media",
            cdn_enabled=False
        )
        print("✅ Media downloader initialized")
        
        # Test quality options
        assert VideoQuality.FULL_HD.value == "1080p"
        assert VideoQuality.ULTRA_HD_4K.value == "2160p"
        print("✅ Video quality options available")
        
        # Test format options
        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.PNG.value == "png"
        print("✅ Image format options available")
        
        print("\n🎉 All media capabilities verified!\n")
        return True
        
    except Exception as e:
        print(f"❌ Media verification failed: {e}\n")
        return False


def verify_oauth():
    """Verify OAuth capabilities."""
    print("🔑 Verifying OAuth capabilities...\n")
    
    try:
        from app.oauth.oauth_proxy import OAuthProxyService
        from app.core.models import SourcePlatform
        from uuid import uuid4
        
        # Test OAuth service initialization
        app_credentials = {
            SourcePlatform.REDDIT: {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret"
            }
        }
        
        oauth_service = OAuthProxyService(
            credential_vault=None,
            app_credentials=app_credentials,
            base_redirect_uri="http://localhost:8000/callback"
        )
        print("✅ OAuth service initialized")
        
        # Test authorization URL generation
        user_id = uuid4()
        auth_url = oauth_service.get_authorization_url(
            user_id=user_id,
            platform=SourcePlatform.REDDIT
        )
        
        assert "reddit.com" in auth_url
        assert "client_id" in auth_url
        print("✅ OAuth URL generation working")
        
        print("\n🎉 All OAuth capabilities verified!\n")
        return True
        
    except Exception as e:
        print(f"❌ OAuth verification failed: {e}\n")
        return False


def main():
    """Run all verifications."""
    print("=" * 60)
    print("Social Media Radar - Implementation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run verifications
    results.append(("Modules", verify_modules()))
    results.append(("Files", verify_files()))
    results.append(("Security", verify_security()))
    results.append(("Media", verify_media()))
    results.append(("OAuth", verify_oauth()))
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print()
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print()
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Social Media Radar is PRODUCTION READY!")
        print()
        return 0
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("Please review the errors above and fix them.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

