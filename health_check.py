# health_check.py
import os
import sys
from pathlib import Path

print("=" * 70)
print("🏥 BACKEND HEALTH CHECK - AI Robotics Book Platform")
print("=" * 70)

# Get current directory
backend_path = Path.cwd()
print(f"\n📂 Backend Path: {backend_path}")

# Check Python version
print(f"\n✅ Python Version: {sys.version.split()[0]}")

# File structure check
print("\n" + "=" * 70)
print("📁 FILE STRUCTURE CHECK")
print("=" * 70)

required_structure = {
    "Root Files": [
        "requirements.txt",
        ".env.example",
        "README.md"
    ],
    "App Directory": [
        "app/__init__.py",
        "app/main.py",
        "app/config.py",
        "app/models.py"
    ],
    "Routes": [
        "app/routes/__init__.py",
        "app/routes/chat.py",
        "app/routes/search.py"
    ],
    "Services": [
        "app/services/__init__.py",
        "app/services/qdrant_service.py",
        "app/services/gemini_service.py"
    ],
    "Utils": [
        "app/utils/__init__.py",
        "app/utils/embeddings.py"
    ]
}

missing_files = []
for category, files in required_structure.items():
    print(f"\n{category}:")
    for file in files:
        file_path = backend_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} - MISSING!")
            missing_files.append(file)

# Environment check
print("\n" + "=" * 70)
print("🔑 ENVIRONMENT VARIABLES CHECK")
print("=" * 70)

env_path = backend_path / ".env"
if env_path.exists():
    print(f"✅ .env file exists ({env_path.stat().st_size} bytes)")
    
    with open(env_path) as f:
        env_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    required_vars = [
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION_NAME",
        "GOOGLE_API_KEY"
    ]
    
    print("\nEnvironment Variables:")
    for var in required_vars:
        found = any(line.startswith(f"{var}=") for line in env_lines)
        if found:
            # Check if it has a value
            value_line = [line for line in env_lines if line.startswith(f"{var}=")][0]
            value = value_line.split("=", 1)[1].strip()
            if value and value != "your_key_here" and value != "":
                print(f"   ✅ {var}: Set ({len(value)} chars)")
            else:
                print(f"   ⚠️  {var}: EMPTY - needs value!")
        else:
            print(f"   ❌ {var}: NOT FOUND!")
else:
    print("❌ .env file NOT FOUND!")
    print("\n💡 Create .env file by copying .env.example:")
    print("   Copy-Item .env.example .env")

# Dependencies check
print("\n" + "=" * 70)
print("📦 PYTHON PACKAGES CHECK")
print("=" * 70)

packages = {
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "qdrant_client": "Qdrant Client",
    "google.generativeai": "Google Generative AI",
    "dotenv": "Python Dotenv",
    "pydantic": "Pydantic",
    "httpx": "HTTPX"
}

missing_packages = []
for package_import, package_name in packages.items():
    try:
        if package_import == "google.generativeai":
            import google.generativeai as genai
            print(f"   ✅ {package_name}: Installed")
        elif package_import == "dotenv":
            from dotenv import load_dotenv
            print(f"   ✅ {package_name}: Installed")
        else:
            module = __import__(package_import)
            version = getattr(module, "__version__", "unknown")
            print(f"   ✅ {package_name}: {version}")
    except ImportError:
        print(f"   ❌ {package_name}: NOT INSTALLED")
        missing_packages.append(package_name)

# Check if main.py is valid
print("\n" + "=" * 70)
print("🔧 MAIN.PY VALIDATION")
print("=" * 70)

try:
    sys.path.insert(0, str(backend_path))
    from app import main
    print("✅ main.py can be imported")
    
    if hasattr(main, 'app'):
        print("✅ FastAPI app object exists")
        
        # Check routes
        if hasattr(main.app, 'routes'):
            routes = [route.path for route in main.app.routes]
            print(f"✅ Routes registered: {len(routes)}")
            for route in routes:
                print(f"   • {route}")
    else:
        print("❌ FastAPI app object not found")
except Exception as e:
    print(f"❌ Error: {e}")

# Final summary
print("\n" + "=" * 70)
print("📊 HEALTH CHECK SUMMARY")
print("=" * 70)

total_issues = len(missing_files) + len(missing_packages)
if not env_path.exists():
    total_issues += 1

if total_issues == 0:
    print("✅ ALL CHECKS PASSED! Backend is healthy! 🎉")
else:
    print(f"⚠️  Found {total_issues} issue(s):")
    if missing_files:
        print(f"   • {len(missing_files)} missing files")
    if missing_packages:
        print(f"   • {len(missing_packages)} missing packages")
    if not env_path.exists():
        print("   • .env file missing")

print("\n" + "=" * 70)