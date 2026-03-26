---
name: Agentic Image pipeline is the gold standard
description: Never deviate from the agentic_image.py pipeline for product image generation — cutout + input_images + "Avoid" prompt is the proven approach. For accessories on models, use the 3-pass pipeline from Product Placement Pro.
type: feedback
---

Always use the agentic_image.py pipeline as-is for product image generation. Do not try to improve it with two-pass, dual references, inpainting, or alternative models.

**Why:** Every attempt to get clever (two-pass with Flux 2 Pro anatomy + Flux 2 Dev shoe, dual input_images, Flux Fill Pro inpainting, Flux 2 Pro image_prompt conditioning) made the shoe worse. The model invents shoes when the reference signal is diluted. The original pipeline already solved the hard problem.

**How to apply:** The pipeline is:
1. Real photo → BRIA bg removal → clean cutout
2. Cutout as `input_images` to Flux 2 Dev (single reference, not dual)
3. Prompt describes ONLY the environment, never the product
4. Append "Avoid: everything but the reference" as negative prompt
5. If the output has anatomy/pose issues, fix the PROMPT (simpler pose, tighter crop) — don't change the pipeline
6. For on-model shots: frame waist/knees-down, feet on the ground, no complex acrobatics
7. Don't make the shoe "the hero, large in frame" — it causes scale distortion. Let the scene breathe.

---

## Accessories on Models — 3-Pass Pipeline (Product Placement Pro)

For purses, tote bags, crossbody bags, backpacks, sandals, and other accessories that need to appear *on a model* with accurate product fidelity, the agentic_image single-reference approach isn't enough — the AI invents its own version of the accessory. The fix is a 3-pass composite pipeline developed in `product-placer-pro/marketing/generate_marketing_v2.py` (Series 5: "Wear It").

**Why:** Flux generates great poses but always invents the product. You can't trust it to preserve a specific bag design, hardware, strap, or colorway. The solution: let the AI handle the *pose*, then swap in the *real product*.

**The 3-Pass Pipeline:**

**Pass 1 — Generate the pose.** Generate a model scene with the AI's version of the product. The only goal is a natural pose (hand position, strap angle, body posture). The AI bag will be thrown away.

**Pass 2 — Remove the AI product.** Inpaint-remove the AI-generated accessory using a removal mask + prompt like "no bag, no backpack, no accessories, clean scene" at high strength (0.95). This gives a clean model in the right pose with empty hands/shoulders.

**Pass 3 — Paste the real product + blend.** Paste the exact product cutout (from the real product photo) onto the clean scene at the correct position, then:
- **3a — Cleanup:** Inpaint around the product to remove any AI bleed-through artifacts
- **3b — Hand-over-product:** Detect skin/hand pixels from the clean scene (RGB skin detection) and layer them *on top* of the pasted product so the model's hand looks like it's gripping the handle naturally
- **3c — Edge blend:** Edge-only inpaint along the product seam for seamless compositing
- **3d — Interaction zones:** Optional inpaint on strap-to-shoulder or handle-to-hand contact areas

**Critical inpainting fixes (hard-won):**
- **Erase before inpaint:** The inpaint model sees original pixels under the mask and reconstructs the same object. Fix: replace masked area with heavy Gaussian blur (radius=40) before sending to FLUX Fill Pro. Removes the visual cue, preserves color/tone context.
- **Mask dilation + feathering:** `MaxFilter(11)` to dilate beyond freehand edges, then `GaussianBlur(5)` for smooth blending. Without this, hard mask edges bleed through.
- **Mask size mismatch:** Frontend canvas can differ from image size. Always resize mask to match image dimensions (`Image.NEAREST`).

**Accessory placement regions (fractional coordinates):**
- `shoulder_bag`: torso right side (0.55, 0.25) → (0.85, 0.60) — crossbody
- `hand_bag`: lower-right beside body (0.60, 0.40) → (0.85, 0.75) — tote/purse
- `backpack`: upper-center back (0.20, 0.10) → (0.80, 0.55) — model facing away
- `feet`: lower portion (0.20, 0.75) → (0.80, 1.00) — shoes/sandals

**Source:** `product-placer-pro/` repo, Series 5 in `marketing/generate_marketing_v2.py`, inpaint fixes in `code/backend/routers/scene.py` and `code/backend/services/tryon_api.py`.
