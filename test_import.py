#!/home/hidara/miniconda3/envs/deim/bin/python
"""
Test if DEIM module can be imported successfully
"""

try:
    # Test main import
    from deim import DEIM
    print("✅ Successfully imported DEIM module")

    # Test initialization with 'under' config
    model = DEIM(config='under')
    print(f"✅ Successfully initialized DEIM with 'under' config")
    print(f"   Device: {model.device}")
    print(f"   Config: {model.config_name}")

    # Test initialization with 'sides' config
    model = DEIM(config='sides')
    print(f"✅ Successfully initialized DEIM with 'sides' config")

    print("\n✨ All import tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")