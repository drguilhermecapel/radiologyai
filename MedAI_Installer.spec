# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['MedAI_CLI_Installer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'threading',
        'json',
        'base64',
        'zipfile',
        'shutil',
        'tempfile',
        'subprocess',
        'winreg',
        'win32com.client',
        'winshell',
        'pathlib',
        'os',
        'sys',
        'time'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'opencv-python',
        'vtk',
        'SimpleITK',
        'nibabel',
        'scikit-image',
        'transformers',
        'timm'
    ],
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
    name='MedAI_Radiologia_Unified_Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
