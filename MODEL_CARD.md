# Model Card — AA-ENet (via `mango-disease-ai` 0.1.3)

Information checked on **2026-10-15** from the PyPI project page and by inspecting the downloaded wheel.

| Item | Value |
|---|---|
| Model | AA-ENet (EfficientNet-B0 backbone + CBAM + 1-layer Transformer encoder), ~5.8M parameters (per package docs) |
| Source | https://pypi.org/project/mango-disease-ai/ · https://github.com/aiub-rd-icca/mango-disease-ai |
| Publisher | AIUB R&D ICCA Research Group (American International University-Bangladesh) |
| License | MIT (package metadata). Architecture code in `backend/app/ml/arch.py` is vendored from it. |
| Wheel SHA-256 (0.1.3) | `aef5d7a63e39b3147882d59b7fc8fe402066f13dd13157e1e76b36e484da405a` (matches PyPI) |
| Weights | `AA-ENet_proposed.pt` (~18.6 MB), bundled inside the wheel; copied to `backend/models/` |
| Task | **Mango LEAF disease classification** (stated in the model's own source docstring). Not a validated fruit classifier. |
| Dataset | "Amropali mango images", 3,500 images, 500 per class (per package docs). Dataset source/license **not independently verified**. |
| Classes (index order) | Anthracnose, Bacterial Canker, Healthy, Powdery Mildew, Scab, Sooty Mould, Stem End Rot |
| Input | RGB, resized to 224×224, ImageNet mean/std normalisation |
| Output | softmax over 7 classes |
| Metrics | **None published; none independently evaluated.** The app reports none. |
| Mango validation | CLIP zero-shot (`openai/clip-vit-base-patch32`, downloaded from Hugging Face on first start, ~600 MB). A preliminary signal only; it can misjudge images. |

## Limitations
- Alpha-stage package from a single maintainer; no linked paper or evaluation found.
- Leaf-trained: photos of fruit or other plant parts are outside the documented scope.
- Domain shift (lighting, camera, background, growth stage) can degrade results.
- Confidence thresholds (0.90 / margin 0.30 / validator 0.50) are conservative defaults, not calibrated on validation data.
- Image classification cannot confirm lab-level or internal infections.
