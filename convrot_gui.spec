# PyInstaller spec for the portable ConvRot Quantizer (one-folder build: torch is far too big for
# --onefile, which would also unpack GBs to %TEMP% on every launch).
#   pyinstaller --noconfirm convrot_gui.spec   ->  dist/ConvRotQuantizer/ConvRotQuantizer.exe
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], ["quant_int8_convrot", "numpy"]  # safetensors save_file needs numpy
for pkg in ("comfy_kitchen", "safetensors"):     # comfy_kitchen loads its backends dynamically
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["convrot_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "IPython", "pytest", "torchvision", "torchaudio", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ConvRotQuantizer",
    console=False,          # GUI app: no console window
    upx=False,              # UPX corrupts torch/CUDA DLLs and trips antivirus
)
coll = COLLECT(exe, a.binaries, a.datas, name="ConvRotQuantizer", upx=False)
