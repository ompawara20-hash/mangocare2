import zipfile, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
skip = {"node_modules", "dist", "__pycache__", ".venv", ".pytest_cache"}
out = root.parent / "MangoCare-AI.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (set(p.relative_to(root).parts) & skip) and p.suffix != ".db" and p.name != ".env":
            z.write(p, pathlib.Path("MangoCare-AI") / p.relative_to(root))
print(out)
