# Image Generation Best Practices

## Prompt Structure

- **Subject first, then context, then style.** "A black cat sitting on a windowsill, rain outside, cinematic photography" — the model builds the image in the order you describe it.
- **Be concrete, not abstract.** "A weathered oak door with peeling green paint and a brass knocker" generates better images than "an old door." Specificity is quality.
- **Use visual language.** Describe what the camera sees, not what you feel. "Golden hour light casting long shadows across cobblestones" not "a beautiful evening."
- **Comma-separate concepts.** Treat the prompt like a tag list for major attributes: subject, setting, lighting, style, quality. Each comma-separated phrase is a distinct instruction.
- **Shorter is often better.** 20-40 words hit the sweet spot for most models. Extremely long prompts dilute attention — the model can't emphasize everything equally.

## Composition & Framing

- **Specify the shot type:**
  - **Close-up / macro** — fills frame with subject, shows fine detail
  - **Medium shot** — subject from waist up, good for portraits with context
  - **Wide / establishing shot** — full environment, subject smaller in frame
  - **Bird's-eye / overhead** — looking straight down
  - **Low angle** — looking up at subject, makes it feel powerful
  - **Dutch angle** — tilted camera, creates unease or energy
- **Depth of field matters.** "Shallow depth of field, bokeh background" isolates the subject. "Deep focus, everything sharp" works for landscapes and architecture.
- **Rule of thirds.** "Subject positioned off-center" or "subject on the left third of the frame" produces more dynamic compositions than dead center.
- **Negative space.** "Minimalist composition with large areas of empty sky" — intentional emptiness creates mood and focus.

## Lighting

- **Name the light source.** "Lit by a single candle," "neon signs reflecting off wet pavement," "overcast diffused daylight." The light defines the mood.
- **Common lighting setups:**
  - **Golden hour** — warm, low-angle sun, long shadows, universally flattering
  - **Blue hour** — cool twilight, moody, cinematic
  - **Rembrandt lighting** — dramatic portrait lighting, triangle of light on cheek
  - **Rim lighting / backlit** — light behind subject, glowing edges, silhouette feel
  - **Flat lighting** — even, no shadows, product photography, editorial
  - **Chiaroscuro** — extreme light/dark contrast, dramatic, painterly
  - **Neon / cyberpunk** — colored artificial light, reflections, night urban scenes
- **Specify light color and direction.** "Warm amber light from the left" is more useful than just "well-lit."
- **Shadows are as important as light.** "Deep shadows," "soft diffused shadows," "harsh midday shadows" each produce radically different images.

## Style & Medium

- **Photography styles:**
  - "Professional photography, DSLR, 85mm lens" — sharp, realistic
  - "Film photography, Kodak Portra 400, grain" — warm, nostalgic
  - "Polaroid, faded colors, white border" — retro instant feel
  - "Street photography, candid, black and white" — documentary
  - "Studio photography, white seamless backdrop" — clean, commercial

- **Art styles:**
  - "Oil painting, impasto brushstrokes" — textured, classical
  - "Watercolor, soft washes, paper texture" — delicate, organic
  - "Digital illustration, flat colors, clean lines" — modern, graphic
  - "Pencil sketch, crosshatching" — raw, conceptual
  - "Anime style, cel shading" — Japanese animation aesthetic
  - "Concept art, matte painting" — epic, cinematic environments

- **Rendering styles:**
  - "3D render, Octane, subsurface scattering" — photorealistic CG
  - "Isometric 3D, low poly" — game art, infographic
  - "Claymation, stop-motion" — tactile, handmade feel
  - "Voxel art, Minecraft style" — blocky, nostalgic

## Quality Modifiers

- **Resolution boosters:** "8K, ultra-detailed, sharp focus, high resolution" — consistently improves detail.
- **Quality keywords:** "Masterpiece, best quality, professional, award-winning" — pushes toward higher quality distributions in the training data.
- **Specificity over superlatives.** "Visible individual hair strands, skin pores, fabric texture" works better than "extremely detailed." Tell the model what kind of detail.
- **Camera/lens references:** "Shot on Hasselblad," "50mm f/1.4 lens," "tilt-shift" — these invoke specific visual characteristics the model has learned.

## Negative Prompts

- **Common negative prompts for photorealism:**
  - "blurry, out of focus, low quality, low resolution"
  - "watermark, text, logo, signature"
  - "deformed, distorted, disfigured, mutation"
  - "extra fingers, extra limbs, missing limbs"
  - "bad anatomy, bad proportions, ugly"
  - "oversaturated, overexposed, underexposed"

- **For illustrations/art:**
  - "photorealistic, photograph" (when you want illustration, not photo)
  - "3D render" (when you want flat/2D)
  - "anime" (when you want Western style)

- **Less is more with negatives.** Long negative prompt lists can confuse the model. Focus on the 3-5 artifacts you most want to avoid.

## Color in Image Generation

- **Name specific colors.** "Teal," "burnt sienna," "dusty rose" generate more accurate colors than "blue-green," "brown," "pink."
- **Specify a palette.** "Color palette: navy, gold, cream" constrains the image to a cohesive look.
- **Monochromatic prompts.** "Monochrome blue, varying shades of navy and sky blue" — forces color harmony.
- **Color temperature sets mood.** "Warm tones, amber, golden" = cozy. "Cool tones, slate, ice blue" = clinical or ethereal.
- **Reference color systems.** "Pantone Living Coral," "Wes Anderson pastel palette," "film noir high contrast black and white" — cultural references work.

## Common Subjects

### People / Portraits
- Specify age, ethnicity, expression, clothing, pose, and gaze direction.
- "Looking directly at camera" vs. "looking away, candid" completely changes the feel.
- Hands are hard. Minimize visible hands or specify "hands in pockets," "hands clasped behind back."
- For realistic faces, add "natural skin texture, subtle imperfections."

### Products
- "Product photography, white background, soft studio lighting, slight reflection on surface."
- Specify the angle: "45-degree angle," "flat lay, top-down," "hero shot, slightly below eye level."
- Include context for lifestyle shots: "coffee mug on a wooden desk next to an open laptop, morning light."

### Landscapes / Environments
- Time of day is critical: "sunrise," "high noon," "dusk," "deep night, moonlit."
- Weather adds mood: "overcast," "foggy," "after rain, wet surfaces reflecting light," "snow falling."
- Scale references: "tiny figure standing at the base of enormous cliffs" establishes grandeur.

### Architecture / Interiors
- Name the style: "mid-century modern," "brutalist concrete," "Victorian Gothic," "Japanese minimalist."
- "Architectural photography, wide-angle lens, converging lines" for exteriors.
- "Interior design magazine photo, staged, natural light through large windows" for interiors.

## Aspect Ratios

- **1:1 (Square)** — Social media posts, profile pictures, icons.
- **4:5 (Portrait)** — Instagram posts, mobile-first content.
- **2:3 / 3:4 (Tall portrait)** — Pinterest, book covers, phone wallpapers.
- **3:2 / 16:9 (Landscape)** — Blog headers, desktop wallpapers, cinematic scenes.
- **21:9 (Ultra-wide)** — Cinematic banners, website hero sections.
- **9:16 (Vertical)** — Stories, Reels, TikTok, mobile full-screen.

## Iteration Strategy

1. **Start broad.** Generate with a simple prompt to see the model's default interpretation.
2. **Refine one thing at a time.** Change lighting OR composition OR style — not all three. You need to know what each change does.
3. **Seed locking.** If a composition works but the style doesn't, lock the seed and change only the style modifiers.
4. **Prompt weighting.** Many models support `(keyword:1.5)` to increase emphasis or `(keyword:0.5)` to decrease. Use sparingly.
5. **Save what works.** Keep a library of effective prompt fragments. "Cinematic lighting, color grading, lens flare" that consistently produces good results can be reused.

## Image Generation Checklist

Before generating, verify:

1. Is the subject clearly described in the first few words?
2. Have you specified composition/framing?
3. Is the lighting defined?
4. Is the style/medium stated?
5. Is the aspect ratio appropriate for the intended use?
6. Have you included quality modifiers?
7. Are negative prompts addressing known failure modes?
8. Is the prompt under 50 words (unless the model specifically benefits from longer prompts)?
