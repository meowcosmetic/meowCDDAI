"""
Script kiểm tra xem các config đã được apply vào code chưa
"""
import re
from pathlib import Path

print("=" * 70)
print("KIỂM TRA CONFIG USAGE")
print("=" * 70)
print()

# Đọc config file để lấy danh sách configs
config_file = Path("text_embeding/gaze_tracking/config.py")
if not config_file.exists():
    print("❌ Không tìm thấy config.py")
    exit(1)

with open(config_file, 'r', encoding='utf-8') as f:
    config_content = f.read()

# Extract config names
config_pattern = r'^\s+([A-Z_]+):\s*[^=]+=\s*[^#]+'
configs = re.findall(config_pattern, config_content, re.MULTILINE)
configs = [c.strip() for c in configs if c.strip() and not c.startswith('#')]

print(f"📋 Tìm thấy {len(configs)} configs trong config.py:")
for cfg in configs[:10]:  # Show first 10
    print(f"   - {cfg}")
if len(configs) > 10:
    print(f"   ... và {len(configs) - 10} configs khác")
print()

# Kiểm tra trong routes_screening_gaze.py
gaze_file = Path("text_embeding/routes_screening_gaze.py")
if not gaze_file.exists():
    print("❌ Không tìm thấy routes_screening_gaze.py")
    exit(1)

with open(gaze_file, 'r', encoding='utf-8') as f:
    gaze_content = f.read()

print("=" * 70)
print("KIỂM TRA TRONG routes_screening_gaze.py:")
print("=" * 70)
print()

# Kiểm tra các config quan trọng
important_configs = [
    'MAX_FRAME_WIDTH',
    'MIN_FOCUSING_DURATION',
    'GAZE_STABILITY_THRESHOLD',
    'OBJECT_DETECTION_INTERVAL',
    'OBJECT_CONFIDENCE_THRESHOLD',
    'OID_MODEL_SIZE',
    'FPS_DEFAULT',
    'LOOKING_AT_OBJECT_THRESHOLD',
    'ADULT_FACE_SIZE_THRESHOLD',
    'CHILD_FACE_SIZE_THRESHOLD',
    'REQUIRE_OBJECT_FOCUS',
    'MIN_OBJECT_FOCUS_RATIO',
    'ALLOW_CAMERA_FOCUS_WITH_ADULT',
    'CAMERA_FOCUS_THRESHOLD',
    'USE_3D_GAZE_CONFIDENCE',
    'MIN_3D_GAZE_CONFIDENCE',
    'ENABLE_WANDERING_DETECTION',
    'BOOK_FOCUSING_SCORE_THRESHOLD',
]

found_configs = []
missing_configs = []
hardcoded_values = []

for cfg in important_configs:
    # Kiểm tra xem config có được sử dụng không
    pattern1 = f'config\\.{cfg}'  # config.MAX_FRAME_WIDTH
    pattern2 = f'config\\.{cfg.lower()}'  # config.max_frame_width
    pattern3 = f'{cfg}'  # MAX_FRAME_WIDTH (có thể là biến local)
    
    if re.search(pattern1, gaze_content, re.IGNORECASE):
        found_configs.append(cfg)
        print(f"✅ {cfg}: Được sử dụng")
    else:
        missing_configs.append(cfg)
        print(f"❌ {cfg}: KHÔNG được sử dụng")
        
        # Kiểm tra xem có hardcoded value không
        if cfg == 'MAX_FRAME_WIDTH':
            if 'max_width = 1280' in gaze_content or 'max_width=1280' in gaze_content:
                print(f"   ⚠️  Tìm thấy hardcoded: max_width = 1280 (nên dùng config.MAX_FRAME_WIDTH)")
                hardcoded_values.append(('MAX_FRAME_WIDTH', '1280'))
        elif cfg == 'FPS_DEFAULT':
            if 'fps = fps if fps > 0 else 30' in gaze_content:
                print(f"   ⚠️  Tìm thấy hardcoded: fps = ... else 30 (nên dùng config.FPS_DEFAULT)")
                hardcoded_values.append(('FPS_DEFAULT', '30'))

print()
print("=" * 70)
print("TÓM TẮT:")
print("=" * 70)
print(f"✅ Configs được sử dụng: {len(found_configs)}/{len(important_configs)}")
print(f"❌ Configs KHÔNG được sử dụng: {len(missing_configs)}/{len(important_configs)}")

if missing_configs:
    print()
    print("📝 Configs cần được thêm vào code:")
    for cfg in missing_configs:
        print(f"   - {cfg}")

if hardcoded_values:
    print()
    print("⚠️  Hardcoded values cần thay thế:")
    for cfg, value in hardcoded_values:
        print(f"   - {cfg}: Tìm thấy hardcoded {value}")

print()
print("=" * 70)




