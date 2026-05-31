# Video Content Best Practices

Use this guide whenever MAUDE creates, edits, renders, publishes, or plans short-form video, HyperFrames compositions, product demos, launch videos, screen recordings, app promos, or social video posts.

## Non-Negotiable Quality Rules

- **Do not rush source footage.** If using a screen wipe, reveal, product demo, app recording, before/after clip, or generated transition video, give the clip enough timeline duration to read. Do not cut away before the visual action completes.
- **Preview motion, not just frames.** Scrub or preview the full rendered timeline before publishing. A still screenshot can look correct while the timing is wrong.
- **No raw URLs in the video frame.** Do not put `https://...`, query strings, app-store URLs, raw download links, or long domains on screen unless the user explicitly asks for a technical demo. Put links in the post/video description instead.
- **Use designed CTA text.** On-screen CTAs should be human-readable phrases such as `Download Pixelus`, `Try Pixelus.io`, `Available on the App Store`, or `Link in description`, not raw URLs.
- **Respect platform behavior.** For YouTube, TikTok, X, LinkedIn, and Instagram, the real clickable link belongs in the post description/caption/profile link field, not burned into the video.
- **Keep visual claims inspectable.** If the video demonstrates an app, product, generated image, or before/after result, hold each important state long enough for a viewer to understand it.

## Timing Guardrails

- App/product reveal clips: minimum 2.0 seconds on screen after the reveal finishes.
- Screen wipe or transition footage: allocate the full source clip duration, or at least 80 percent of it if intentionally trimmed.
- Before/after comparisons: hold before 1.0-1.5 seconds, transition 0.8-1.5 seconds, after 2.0-3.0 seconds.
- Text cards: 2.0-3.5 seconds for one short line; 3.5-5.0 seconds for two lines. Never animate text so fast it cannot be read aloud.
- Final CTA: hold 2.0-3.0 seconds. Keep it simple and designed.

## Composition Rules

- Start with a shot list and timeline table before rendering: scene, purpose, asset, start time, duration, on-screen text, narration/caption, and transition.
- Treat generated wipe/reveal videos as primary footage, not decorative B-roll. Their start, midpoint, and end need to be visible.
- Avoid stacking too many animations at once. If the product is moving, text should be stable; if text is animating, product footage should be calm.
- Use safe margins for vertical video. Keep important text and UI away from the top/bottom platform chrome areas.
- Use one clear message per scene. Do not mix a feature explanation, URL, CTA, and demo motion in the same second.

## Publishing Rules

- Every publish plan must include: video title, caption/description, designed CTA, actual link placement, tags/hashtags, and privacy/platform setting.
- Put the real app/site URL in the description/caption field. For Pixelus, use a clean link such as `https://pixelus.io` in the description unless the user provides a different URL.
- Before posting, verify that media is attached and that the caption/description contains the link when a link is needed.
- Before any `youtube_upload` or `social_post` with video, call `video_pre_publish_checklist` and create the checklist artifact. Do not publish if the checklist status is BLOCKED.
- Do not publish if the video contains a visible placeholder, raw URL, broken asset, missing media, unreadable text, or clipped transition.

## Required Checklist Artifact

Before publishing or posting a video, create a markdown pre-publish checklist artifact with `video_pre_publish_checklist`. The artifact must include:

- Final title
- Final description/caption
- Link URL and link placement
- Designed on-screen CTA text
- Tags/hashtags
- Privacy/platform setting
- Timing notes for transitions, wipes, reveals, and important holds
- Render review notes from watching or scrubbing the actual MP4
- Explicit pass/fail checks for media attachment, readable text, placeholders, raw URLs, and link placement

If any check fails, fix the video or metadata first, then create a new checklist.

## Review Checklist Before Render or Publish

1. Does every transition or screen wipe have time to complete?
2. Can a viewer understand each important app/product state without pausing?
3. Are all on-screen CTAs designed phrases rather than raw links?
4. Is the real clickable link in the description/caption, not only in the video?
5. Does the final render play through without timing, cropping, or audio issues?
6. Did you inspect the actual MP4, not just the HTML/source files?
