# Web Design Patterns & UI/UX

## UI Component Patterns

### Modals & Dialogs
- **Use modals sparingly.** They interrupt flow. If the content can live inline or on its own page, don't put it in a modal.
- **Always provide a close mechanism.** X button, click-outside-to-close, and Escape key. All three. Missing any one frustrates users.
- **Focus trap inside modals.** Tab should cycle through modal elements only, not reach the background page. Return focus to the trigger element on close.
- **Confirmation modals for destructive actions only.** "Are you sure you want to delete?" — yes. "Are you sure you want to save?" — no, just save.
- **Size modals to content.** Don't make every modal full-screen. Small confirmations get small modals. Complex forms get larger ones. Never scroll the modal and the background simultaneously.

### Toasts & Notifications
- **Auto-dismiss success toasts (3-5 seconds).** Users don't need to manually close "Saved successfully."
- **Persist error toasts until dismissed.** Errors need time to read and act on. Don't auto-hide them.
- **Stack from the bottom or top, consistently.** New toasts push old ones, not replace them. Limit to 3 visible at once.
- **Include an action when useful.** "File deleted. [Undo]" is more helpful than just "File deleted."
- **Don't use toasts for critical errors.** If the user can't proceed without seeing the message, use an inline error or modal — toasts are too easy to miss.

### Loading States
- **Skeleton screens over spinners.** Skeletons show the shape of incoming content, giving users a sense of structure. Spinners say nothing about what's loading.
- **Skeleton rules:** Match the layout of the actual content. Use subtle pulse animation. Gray blocks for text, rounded rectangles for images. Never show skeletons for more than 3 seconds — if it takes longer, add a progress indicator.
- **Spinners for actions, skeletons for pages.** Button click → spinner inside the button. Page load → skeleton of the layout.
- **Optimistic UI for fast actions.** When the user toggles a like or saves a setting, update the UI immediately and reconcile with the server in the background. Roll back on failure.
- **Progress bars for known-duration tasks.** File uploads, imports, builds. Show percentage and estimated time remaining. Indeterminate progress bars are better than nothing but worse than real progress.
- **Disable the trigger during loading.** A submit button should show a spinner and be disabled to prevent double-submission. Never leave it clickable.

### Empty States
- **Empty states are onboarding opportunities.** "No projects yet" is a dead end. "No projects yet. Create your first project to get started. [Create Project]" guides the user.
- **Include illustration or icon.** A visual makes empty states feel intentional, not broken.
- **Provide the primary action.** The CTA to create/add/import the first item should be the most prominent element.
- **Differentiate between "no results" and "empty."** Search returning nothing ("No results for 'xyz'. Try a different search term.") is different from a genuinely empty collection.

### Infinite Scroll vs. Pagination
- **Infinite scroll for browsing.** Social feeds, image galleries, discovery interfaces. When the user has no specific target, let them keep scrolling.
- **Pagination for reference.** Search results, admin tables, data lists. When users need to return to a specific page or share a link to page 3, pagination is essential.
- **Always show a loading indicator at the scroll threshold.** Don't just silently append content — the user should know more is coming.
- **Provide a "back to top" button** with infinite scroll. After 5+ screen-lengths of scrolling, it's essential.
- **Virtual scrolling for large lists.** Render only visible rows in the DOM. Lists with 10,000+ items will destroy performance without virtualization (react-window, TanStack Virtual).

### Drag and Drop
- **Visual affordance.** Draggable items need a grip handle (⠿) or cursor change. Don't make the user guess what's draggable.
- **Show the drop zone clearly.** Highlight valid drop targets with a border, background change, or placeholder. Dim invalid zones.
- **Support keyboard alternatives.** Not all users can drag. Provide arrow-key reordering or a "move to" menu.
- **Smooth animations on drop.** Items should animate into their new position, not teleport. 150-200ms transition.

## Form Design Patterns

### Input Types
- **Use the right HTML input type.** `type="email"` gives mobile users an @ keyboard. `type="tel"` gives a number pad. `type="url"` includes protocol keys. Small detail, big impact.
- **Date pickers for dates, not text inputs.** Native `<input type="date">` works in all modern browsers. Custom pickers for complex date ranges.
- **Toggles for binary settings, checkboxes for consent.** "Enable dark mode" → toggle. "I agree to terms" → checkbox. Don't mix them.
- **Dropdowns for 5-15 options.** Fewer than 5 → radio buttons. More than 15 → searchable combobox/autocomplete.
- **Textarea auto-grow.** Let textareas expand with content instead of forcing a fixed height with scrollbar. Set a max-height for sanity.

### Validation
- **Validate on blur, not on every keystroke.** Checking email format while the user is still typing is annoying. Check when they leave the field.
- **Exception: password strength.** Show strength feedback in real-time as they type. This is the one case where keystroke validation helps.
- **Inline errors below the field.** Red text, red border on the field. Don't put all errors at the top of the form — users can't connect them to the right field.
- **Mark required fields, not optional ones.** Most fields are required in a well-designed form. Mark the minority (optional fields) with "(optional)" text.
- **Preserve input on error.** Never clear the form on validation failure. The user shouldn't have to re-enter everything because one field was wrong.

### Multi-Step Forms (Wizards)
- **Show progress.** Step indicator at the top: "Step 2 of 4" with a progress bar or numbered steps.
- **Allow going back without losing data.** Every previous step should be revisitable with data intact.
- **Validate per step, not all at the end.** Don't let a user reach step 4 only to find out step 1 had an error.
- **Summary step before submission.** For important workflows (checkout, applications), show a review page with all entered data before the final submit.
- **Save progress for long forms.** If a form takes more than 2 minutes, auto-save drafts. Let users return later.

## Navigation Patterns

### Sidebar Navigation
- **Collapsible on desktop, hidden on mobile.** Use a hamburger or overlay for mobile. On desktop, allow collapsing to icon-only for more content space.
- **Group related items.** Use section headers (not just dividers) to organize nav items into logical groups.
- **Active state must be obvious.** Background highlight, left border accent, bold text — the user must know exactly where they are.
- **Limit depth to 2 levels.** Primary items with one level of sub-items. Deeper nesting means your information architecture needs restructuring.

### Tab Navigation
- **Tabs for same-level views of the same data.** "Overview | Activity | Settings" for a project page. Not for unrelated pages.
- **Don't use tabs for sequential steps.** That's a wizard/stepper. Tabs imply any-order access.
- **Highlight the active tab.** Underline, background change, or border. The active tab should look connected to its content panel.
- **Keep tab labels short.** One or two words. If labels need sentences, tabs are the wrong pattern.
- **Persist tab state in the URL.** `?tab=settings` so users can bookmark or share specific tabs.

### Breadcrumbs
- **Use for hierarchical navigation.** Home > Category > Subcategory > Item. Not for multi-step processes (that's a stepper).
- **Last item is the current page.** Display it as plain text (not a link). All previous items are links.
- **Don't use breadcrumbs as the only navigation.** They supplement a sidebar or top nav, not replace it.

### Command Palette
- **Cmd+K / Ctrl+K is the standard trigger.** Users expect this in modern web apps. It's fast for power users who know what they want.
- **Fuzzy search.** Match on partial strings, out-of-order words, abbreviations. "usr set" should find "User Settings."
- **Show keyboard shortcuts next to actions.** The command palette is where users discover shortcuts.
- **Recent items first.** Show recently used commands/pages before alphabetical listing.

## Data Display Patterns

### Tables
- **Sticky header row.** When scrolling long tables, the header must remain visible.
- **Sortable columns.** Click header to sort. Show sort direction with an arrow icon. Default sort should be the most useful (usually most recent first).
- **Row hover highlight.** Subtle background change on hover helps track across wide tables.
- **Responsive tables:** On mobile, either horizontal scroll with a frozen first column, or reflow into card layout. Never let a table break the page width.
- **Bulk actions.** Checkbox column + action bar that appears when items are selected. "3 selected: [Delete] [Export] [Archive]"
- **Empty column handling.** Show a dash (—) not blank. Blanks look like missing data; dashes are intentional "no value."

### Cards
- **One primary action per card.** The whole card can be clickable, or have one clear CTA button. Don't put 5 buttons on a card.
- **Consistent card height in grids.** Use `min-height` or fixed aspect ratios. Jagged card grids look broken.
- **Image at top, content below.** The standard card layout works because it mirrors how we read: visual → headline → details → action.
- **Truncate long content with ellipsis.** Show first 2-3 lines with "Read more" or expand-on-click. Don't let one card push the grid out of alignment.

### Dashboards
- **Most important metric, top-left.** Users scan in an F-pattern. Put the key number where eyes land first.
- **Limit to 5-7 widgets.** More than that and nothing has emphasis. Every widget should answer a specific question.
- **Date range selector.** Dashboards without time context are useless. Default to last 7 or 30 days. Let users customize.
- **Comparison context.** "Revenue: $42,000" means nothing alone. "Revenue: $42,000 (+12% vs last month)" tells a story.
- **Loading states per widget.** Load each widget independently. Don't block the entire dashboard while one slow query runs.

## CSS Architecture

### Naming Conventions
- **BEM for vanilla CSS.** `.block__element--modifier`. Verbose but unambiguous. `.card__title--highlighted`.
- **Utility-first with Tailwind.** `class="flex items-center gap-4 p-6 rounded-lg shadow"`. Fast to build, readable once you know the utilities.
- **CSS Modules or scoped styles for components.** Auto-generated unique class names prevent collisions. Best for React/Vue/Svelte component architecture.
- **Pick one system and be consistent.** Mixing BEM, Tailwind, and inline styles in the same project creates unmaintainable chaos.

### Design Tokens
- **Define tokens, not raw values.** `--color-primary: #2563eb`, `--spacing-md: 1rem`, `--radius-lg: 0.75rem`. Reference tokens everywhere, raw values nowhere.
- **Semantic tokens over literal ones.** `--color-error` not `--color-red`. `--spacing-section` not `--spacing-48`. Meaning is more useful than description.
- **Store tokens in CSS custom properties.** `var(--color-primary)` works everywhere, supports runtime theming (dark mode toggle), and needs no build step.
- **Scale tokens systematically.** Spacing: 4, 8, 12, 16, 24, 32, 48, 64. Font sizes: 12, 14, 16, 18, 20, 24, 30, 36. Consistent scales prevent arbitrary values.

### Component Structure
- **Co-locate styles with components.** `Button.tsx` + `Button.module.css` in the same directory. Don't hunt through a global stylesheet for component styles.
- **Build small, compose big.** `<Button>`, `<Input>`, `<Avatar>` → `<UserCard>` → `<UserList>` → `<DashboardPanel>`. Each level is self-contained.
- **Prop-driven variants, not CSS overrides.** `<Button variant="primary" size="lg">` not `.custom-button { override everything }`. Components should expose their own API.
- **Responsive from the inside out.** Components should be responsive to their container, not the viewport. Use container queries (`@container`) when possible.

## Interaction & Motion

### Transitions
- **150-300ms for UI transitions.** Shorter feels instant, longer feels sluggish. 200ms is the sweet spot for most interactions.
- **Ease-out for entrances, ease-in for exits.** Elements entering the screen should decelerate (ease-out). Elements leaving should accelerate (ease-in). Ease-in-out for position changes.
- **Don't animate layout shifts.** Animating width/height triggers reflow. Animate `transform` and `opacity` — they're GPU-accelerated and don't cause layout recalculation.
- **Respect `prefers-reduced-motion`.** Some users get motion sick. Wrap animations in `@media (prefers-reduced-motion: no-preference)`.

### Micro-interactions
- **Button feedback.** Press effect (scale down slightly), color change, or ripple on click. The user needs to know their click registered.
- **Toggle animations.** Switches should slide, not teleport. Checkboxes should fill with a quick animation. 150ms.
- **Hover reveals.** Show secondary actions (edit, delete icons) on card hover. Keeps the default view clean while making actions discoverable.
- **Scroll-triggered animations.** Fade-in as elements enter the viewport. Use `IntersectionObserver`, not scroll event listeners. Keep animations subtle — flying elements are distracting.

## Dark Mode

- **Don't just invert colors.** Dark mode needs its own palette. Pure white text on pure black background causes halation (glowing edges). Use off-white (#E0E0E0) on dark gray (#1A1A1A).
- **Reduce saturation in dark mode.** Fully saturated colors on dark backgrounds vibrate and cause eye strain. Desaturate by 10-20%.
- **Elevate with lighter surfaces, not shadows.** In dark mode, shadows are invisible. Use progressively lighter background shades to show elevation: base (#121212), card (#1E1E1E), modal (#2A2A2A).
- **Test both modes equally.** Don't design in light mode and port to dark as an afterthought. Test every component, every state, every illustration in both modes.
- **Use CSS custom properties for theming.** Toggle a class on `<html>` that swaps the entire token set. `html[data-theme="dark"] { --bg: #121212; --text: #E0E0E0; }`.
- **Persist the preference.** Save to `localStorage`. Default to `prefers-color-scheme` media query on first visit.

## Web Design Checklist

Before shipping a feature, verify:

1. Does every loading state have visual feedback (skeleton, spinner, or progress)?
2. Are error states helpful, not just red text?
3. Do empty states guide the user toward action?
4. Is every modal closeable via X, Escape, and click-outside?
5. Do forms validate on blur with inline error messages?
6. Is the navigation state visible (active page, breadcrumbs)?
7. Are transitions under 300ms and respecting reduced-motion preference?
8. Does dark mode look intentionally designed, not auto-inverted?
9. Are tables sortable, with sticky headers and responsive behavior?
10. Can a keyboard-only user complete every workflow?
