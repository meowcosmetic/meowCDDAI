#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra xem librosa, soundfile, moviepy đã được cài đặt chưa
"""

import sys

print("=" * 50)
print("Kiểm tra cài đặt Speech Analysis Modules")
print("=" * 50)
print()

# Test librosa
try:
    import librosa
    print(f"✅ librosa: {librosa.__version__}")
except ImportError as e:
    print(f"❌ librosa: CHƯA CÀI ĐẶT")
    print(f"   Error: {e}")
    sys.exit(1)

# Test soundfile
try:
    import soundfile
    print(f"✅ soundfile: {soundfile.__version__}")
except ImportError as e:
    print(f"❌ soundfile: CHƯA CÀI ĐẶT")
    print(f"   Error: {e}")
    sys.exit(1)

# Test moviepy
try:
    import moviepy
    print(f"✅ moviepy: {moviepy.__version__}")
except ImportError as e:
    print(f"❌ moviepy: CHƯA CÀI ĐẶT")
    print(f"   Error: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("🎉 Tất cả modules đã được cài đặt thành công!")
print("=" * 50)
print()
print("Bây giờ bạn có thể:")
print("  1. Chạy server: python main.py")
print("  2. Hoặc dùng: run_server_python312.bat")
print("  3. Test API: python test_speech_api.py")

