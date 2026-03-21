# Website Design Best Practices

## Layout & Structure

- **Visual hierarchy is everything.** The most important content should be the most visually prominent. Use size, weight, color, and spacing to guide the eye.
- **Give text room to breathe.** Use generous line-height (1.7-1.8), ample margins between headers, paragraphs, tags, and links. Cramped text feels cheap and is hard to read.
- **Consistent grid system.** Align elements to a grid. Inconsistent alignment looks amateur even if the individual elements are well-designed.
- **Limit content width.** Body text should be 60-75 characters per line (roughly 600-800px). Full-width text is exhausting to read.
- **Whitespace is a design element.** Don't fill every pixel. Negative space creates focus, elegance, and clarity.
- **F-pattern and Z-pattern.** Users scan in predictable patterns. Place key content (CTAs, navigation, headlines) along these scan paths.

## Typography

- **Two fonts maximum.** One for headings, one for body. More than two creates visual noise.
- **Body text: 16px minimum.** On screens, smaller than 16px causes readability issues and forces users to zoom.
- **Establish a type scale.** Use a consistent ratio (1.25x, 1.333x, 1.5x) for heading sizes. Don't pick arbitrary numbers.
- **Font weight for hierarchy.** Use weight (400, 600, 700) to create contrast, not just size. A bold 18px heading can outperform a thin 24px heading.
- **Line-height: 1.5-1.8 for body text.** Tighter (1.2-1.3) for headings. Never use the default 1.0.
- **Limit line length.** Use `max-width` on text containers. Nobody wants to read a 200-character line.

## Color

- **Start with a limited palette.** One primary color, one accent, neutrals. Add color only when it serves a purpose.
- **Contrast ratios matter.** WCAG AA requires 4.5:1 for body text, 3:1 for large text. Test with a contrast checker.
- **Don't rely on color alone.** Color-blind users exist. Use icons, patterns, or labels alongside color indicators.
- **Dark mode is expected.** Design for both light and dark. Use CSS custom properties to make theming manageable.
- **Consistent color semantics.** Red = error/danger, green = success, yellow = warning. Don't repurpose these.

## Navigation

- **Keep it simple.** 5-7 top-level navigation items maximum. If you need more, your information architecture needs work.
- **Users should always know where they are.** Active states, breadcrumbs, highlighted nav items. Never leave the user guessing.
- **Logo links to home.** Always. It's a universal convention.
- **Mobile nav: hamburger is fine.** Users understand it. Don't reinvent navigation patterns.
- **Footer navigation is valuable.** Put secondary links, legal pages, and sitemap links in the footer. Users scroll.

## Responsive Design

- **Mobile-first.** Design for the smallest screen first, then enhance for larger screens. It's easier to add than to remove.
- **Breakpoints based on content, not devices.** Resize until the design breaks, then add a breakpoint there. Don't target specific phone models.
- **Touch targets: 44x44px minimum.** Fingers are bigger than cursors. Small tap targets frustrate mobile users.
- **Test on real devices.** Browser DevTools are approximations. Real phones have different rendering, different scroll behavior, different performance.
- **Images must be responsive.** Use `srcset` and `sizes`. Serving a 2MB hero image to a phone on 3G is hostile.

## Performance

- **Page load under 3 seconds.** Users leave after 3 seconds. Aim for under 1.5 on decent connections.
- **Optimize images.** Use WebP/AVIF, compress aggressively, lazy-load below-the-fold images. Images are typically 50%+ of page weight.
- **Minimize JavaScript.** Every KB of JS blocks rendering. Question whether you need that library or if CSS/HTML can do it.
- **Use system fonts as fallback.** Custom fonts are fine, but ensure text is visible immediately (use `font-display: swap`).
- **Core Web Vitals matter.** LCP (loading), FID/INP (interactivity), CLS (visual stability). Google ranks on these.

## Accessibility

- **Semantic HTML first.** Use `<nav>`, `<main>`, `<article>`, `<button>`, `<header>`. Divs with click handlers are not buttons.
- **Alt text on all images.** Descriptive for content images, empty (`alt=""`) for decorative ones.
- **Keyboard navigation must work.** Tab through your entire site. Can you reach and activate everything without a mouse?
- **Form labels are mandatory.** Every input needs a visible `<label>`. Placeholder text is not a label.
- **Skip navigation link.** Screen reader users shouldn't have to tab through your entire nav on every page.
- **ARIA only when HTML can't do it.** Native HTML elements have built-in accessibility. ARIA is a supplement, not a replacement.

## Content

- **Above the fold: answer "what is this?"** Within 5 seconds, a new visitor should understand what the site/product does and why they should care.
- **Scannable content.** Use headings, bullet points, bold key phrases. Nobody reads walls of text on the web.
- **One CTA per section.** Don't compete with yourself. Each section should drive toward one action.
- **Write for humans, not SEO bots.** Good content that answers real questions will rank. Keyword-stuffed garbage won't convert even if it ranks.
- **Social proof near CTAs.** Testimonials, stats, logos. Place them where decisions are made.

## Forms

- **Ask for the minimum.** Every additional field reduces completion rates. Do you really need their phone number?
- **Inline validation.** Show errors as users type, not after they submit. Don't make them hunt for what went wrong.
- **Clear error messages.** "Invalid input" is useless. "Email must include an @ symbol" is helpful.
- **Smart defaults.** Pre-fill what you can (country from IP, date format from locale).
- **Progress indicators for multi-step forms.** "Step 2 of 4" reduces abandonment.

## Design System Checklist

Before launching, verify:

1. Does it look good on phone, tablet, and desktop?
2. Is the text readable without zooming?
3. Do all interactive elements have hover/focus/active states?
4. Does it load in under 3 seconds on a mid-range phone?
5. Can you navigate the entire site with just a keyboard?
6. Is the color contrast accessible?
7. Do all images have appropriate alt text?
8. Is the CTA obvious on every page?
