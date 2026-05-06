# Color Theory Guide

## The Color Wheel

- **Primary colors (RYB):** Red, yellow, blue. Cannot be mixed from other colors. The foundation of all color mixing in traditional art and design.
- **Primary colors (RGB):** Red, green, blue. Used for screens and digital displays. Light-based — mixing all three at full intensity produces white.
- **Primary colors (CMYK):** Cyan, magenta, yellow, black. Used for print. Ink-based — mixing all produces near-black (K is added for true black).
- **Secondary colors:** Created by mixing two primaries. Orange (red+yellow), green (yellow+blue), violet (red+blue).
- **Tertiary colors:** Created by mixing a primary with an adjacent secondary. Red-orange, yellow-green, blue-violet, etc. These are the "in-between" colors.

## Color Properties

### Hue
The pure color itself — red, blue, green, orange. What most people mean when they say "color." Hue is position on the color wheel.

### Saturation (Chroma)
How pure or intense the color is. Full saturation = vivid, electric. Low saturation = muted, grayish. Desaturated colors feel sophisticated; oversaturated colors feel loud.

### Value (Lightness/Brightness)
How light or dark the color is. Adding white creates a **tint**. Adding black creates a **shade**. Adding gray creates a **tone**. Value contrast is more important than hue contrast for readability and hierarchy.

## Color Harmony Systems

### Complementary
Colors opposite each other on the wheel (e.g., blue and orange). Maximum contrast, high energy. Great for CTAs and emphasis. Can feel jarring if overused — use one as dominant and the other sparingly.

### Analogous
Three colors adjacent on the wheel (e.g., blue, blue-green, green). Harmonious and pleasing. Low contrast — serene, natural feel. Risk: can feel monotonous without a value range.

### Triadic
Three colors equally spaced on the wheel (e.g., red, yellow, blue). Vibrant and balanced. Works well when one dominates and the other two accent. Comic books and superhero branding use this frequently.

### Split-Complementary
One base color plus the two colors adjacent to its complement (e.g., blue + red-orange + yellow-orange). Similar contrast to complementary but less tension. Easier to work with, still visually rich.

### Tetradic (Double Complementary)
Four colors — two complementary pairs (e.g., blue+orange, red+green). Complex and rich. Hard to balance — let one color dominate, use the others as accents.

### Monochromatic
One hue in varying tints, shades, and tones. Clean, elegant, cohesive. Add contrast through value (light/dark) not hue. Risk: can feel flat without enough value range.

## Color Temperature

- **Warm colors:** Red, orange, yellow. Advance toward the viewer, feel energetic, urgent, inviting. Use for CTAs, alerts, emphasis.
- **Cool colors:** Blue, green, violet. Recede from the viewer, feel calm, professional, spacious. Use for backgrounds, body text, secondary elements.
- **Temperature is relative.** A yellow-green feels warm next to blue, but cool next to orange. Context changes perception.
- **Warm light, cool shadow.** In natural lighting, illuminated areas are warm and shadows are cool. Reversing this feels unnatural (which can be a deliberate creative choice).

## Color Psychology

### Red
Energy, urgency, danger, passion, appetite. Raises heart rate. Used for clearance sales, food brands, stop signs. Can mean danger or love depending on context.

### Orange
Enthusiasm, creativity, warmth, affordability. Friendlier than red, less aggressive. Call-to-action buttons, playful brands, food industry.

### Yellow
Optimism, attention, caution, intellect. Most visible color — used for warning signs. Can feel cheerful or anxious depending on shade. Hard to read as text on white.

### Green
Nature, growth, health, money, safety. Easiest on the eye. Works for health brands, finance, environmental, "go" signals. Dark green = wealth/prestige. Bright green = eco/organic.

### Blue
Trust, stability, professionalism, calm. Most universally liked color. Dominates tech, finance, healthcare, corporate. Cool, reliable, but can feel cold or impersonal.

### Violet/Purple
Luxury, creativity, mystery, spirituality. Historically associated with royalty (expensive dye). Premium brands, creative industries, beauty products.

### Black
Power, elegance, sophistication, authority. Luxury brands, fashion, formal contexts. Can feel heavy or oppressive in large amounts.

### White
Purity, simplicity, cleanliness, space. Apple's brand strategy. Creates breathing room. "White space" is a design tool, not emptiness.

## Practical Application

### The 60-30-10 Rule
- **60%** — Dominant color (usually a neutral or subdued tone). Walls, backgrounds, large surfaces.
- **30%** — Secondary color. Furniture, panels, supporting elements.
- **10%** — Accent color. CTAs, highlights, small pops. This is where your bold color lives.

### Contrast for Readability
- Dark text on light background: minimum 4.5:1 contrast ratio (WCAG AA).
- Light text on dark background: same ratio, but be careful — thin white text on dark backgrounds can feel harsh. Use slightly off-white (#F0F0F0) on slightly off-black (#1A1A1A).
- Never put saturated text on saturated background of a different hue. Red text on blue background is physically painful to read (chromatic aberration).

### Building a Palette from Scratch
1. **Start with one color** — your brand color or the mood you want.
2. **Choose a harmony system** — complementary for energy, analogous for calm, triadic for vibrancy.
3. **Add neutrals** — warm grays (brownish) or cool grays (bluish) depending on your palette temperature.
4. **Create tints and shades** — 5-7 steps from near-white to near-black for your primary color. This gives you your full working palette.
5. **Assign semantic roles** — primary, secondary, accent, background, text, success, warning, error.
6. **Test in context** — colors look different surrounded by other colors. Always evaluate the full composition, not swatches in isolation.

### Common Mistakes
- **Too many colors.** 3-5 colors (plus neutrals) is enough for any project. More creates chaos.
- **Equal amounts of every color.** Without a dominant color, nothing anchors the design. Use the 60-30-10 rule.
- **Ignoring value contrast.** Two colors can have different hues but identical values — they'll blur together. Test in grayscale.
- **Pure black (#000000) for text.** On screens, pure black on pure white creates maximum contrast that causes eye strain. Use dark gray (#1A1A1A to #333333) instead.
- **Neon/fully saturated palettes.** Fully saturated colors vibrate against each other and cause eye fatigue. Desaturate slightly for professional results.
- **Not testing for color blindness.** ~8% of men are color blind. Use tools like Coblis or Sim Daltonism to verify your palette works without full color perception.

## Digital Color Spaces

### Hex (#RRGGBB)
6-digit hexadecimal. Web standard. `#FF6600` = red:255, green:102, blue:0 (orange). Shorthand: `#F60`.

### RGB (0-255)
Direct screen color. `rgb(255, 102, 0)`. Good for code, but hard to intuit relationships between colors.

### HSL (Hue, Saturation, Lightness)
`hsl(24, 100%, 50%)`. Much more intuitive than RGB. Hue = position on wheel (0-360), Saturation = intensity (0-100%), Lightness = light/dark (0-100%). To create tints: increase lightness. To create shades: decrease lightness. To mute: decrease saturation.

### OKLCH (Perceptually Uniform)
Modern CSS color space. `oklch(70% 0.15 50)`. Perceptually uniform — changing lightness by the same amount looks like the same visual change across all hues. Better than HSL for generating consistent palettes. Supported in modern browsers.

## Quick Reference: Harmony Cheat Sheet

| Mood | Harmony | Example |
|------|---------|---------|
| Bold, high-energy | Complementary | Blue + Orange |
| Calm, natural | Analogous | Green + Teal + Blue |
| Vibrant, playful | Triadic | Red + Yellow + Blue |
| Rich, nuanced | Split-complementary | Purple + Yellow-green + Yellow-orange |
| Elegant, minimal | Monochromatic | Navy + Blue + Sky + Ice |
| Professional, safe | Blue + neutral grays | Corporate standard |
| Warm, inviting | Analogous warm | Terracotta + Amber + Cream |
