# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../models', 'models'),
        ('../data', 'data'),
    ],
    hiddenimports=[
        'tensorflow',
        'pydicom',
        'cv2',
        'PIL',
        'PyQt5',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'h5py',
        'SimpleITK',
        'nibabel',
        'skimage',
        'reportlab',
        'cryptography',
        'pyqtgraph',
        'vtk',
        'transformers',
        'timm',
        'click',
        'jinja2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    icon='../icons/medai_icon.ico'
)
