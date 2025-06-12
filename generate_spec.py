import PyInstaller.__main__
import os

spec_content = '''

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('assets', 'assets'),
        ('templates', 'templates'),
    ],
    hiddenimports=[
        'tensorflow',
        'tensorflow.python.keras.engine.base_layer_v1',
        'pydicom',
        'pydicom.encoders',
        'cv2',
        'PIL',
        'sklearn',
        'flask',
        'flask_cors',
        'SimpleITK',
        'nibabel',
        'skimage',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MedAI_Radiologia',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)
'''

with open('medai_radiologia.spec', 'w') as f:
    f.write(spec_content)

print("✅ Arquivo spec gerado com sucesso!")
