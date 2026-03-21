# Writing Best Practices

## Clarity Above All

- **Say what you mean in the fewest words possible.** If a sentence works without a word, cut the word. "In order to" is just "to." "At this point in time" is just "now."
- **One idea per sentence.** Long compound sentences with multiple clauses force readers to hold too much in working memory. Split them.
- **One topic per paragraph.** Each paragraph should make one point. Start with the point, then support it. If the topic shifts, start a new paragraph.
- **Active voice by default.** "The team shipped the feature" not "The feature was shipped by the team." Passive voice hides the actor and weakens the sentence.
- **Avoid weasel words.** "Arguably," "somewhat," "fairly," "rather," "quite" — these hedge without adding information. Either commit to the claim or cut it.
- **Define jargon or don't use it.** If your audience might not know a term, explain it the first time. If you can say it in plain language, do that instead.

## Structure & Organization

- **Lead with the conclusion.** Don't bury the point at the end. State your main idea first, then provide evidence and context. Readers may not finish.
- **Use headings to create a map.** A reader should be able to scan just the headings and understand the document's structure and key points.
- **Bullet points for lists, prose for arguments.** Lists work when items are parallel and scannable. Complex reasoning needs full sentences and logical flow.
- **Short paragraphs.** On screen, 3-4 sentences per paragraph is ideal. A wall of text is a wall that readers hit and bounce off.
- **Transitions signal direction.** "However" signals a turn. "Additionally" signals more of the same. "Therefore" signals a conclusion. Use them to guide the reader.
- **Consistent structure across sections.** If section one has overview > details > example, keep that pattern throughout. Readers learn your structure and move faster.

## Tone & Voice

- **Write like you talk, then tighten it up.** Conversational writing is engaging. But remove the ums, the tangents, the run-ons. Keep the warmth, cut the slack.
- **Match the audience.** Technical writing for engineers can assume context and use domain terms. Writing for general audiences cannot. Know who you're writing for.
- **Confident, not arrogant.** State things directly. "This approach works because..." not "I humbly believe that perhaps this approach might possibly work because..."
- **Be specific, not vague.** "Response times improved by 40%" not "Response times significantly improved." Numbers, names, and concrete details build credibility.
- **Avoid corporate speak.** "Leverage synergies to drive holistic solutions" means nothing and everyone knows it. Say what you actually mean.

## Technical Writing

- **Explain the why before the how.** Context makes instructions meaningful. "To prevent data loss during deploys, we run migrations before swapping containers" is better than just the command.
- **Code examples are worth a thousand words.** Show, don't just tell. A working example answers questions that prose can't anticipate.
- **Step-by-step instructions must be tested.** Follow your own instructions on a clean setup. If you skip a step because "everyone knows that," your reader doesn't.
- **Error messages are writing.** They should say what went wrong, why, and what to do about it. "Error: Connection refused" is a fact. "Error: Cannot connect to database at localhost:5432. Is PostgreSQL running?" is helpful.
- **Keep docs near the code.** Documentation that lives far from the code it describes drifts and becomes unreliable. READMEs, inline comments, and docstrings stay current because they're visible.

## Editing & Revision

- **First draft = get ideas down. Second draft = get them right.** Don't try to write perfectly on the first pass. Write freely, then revise ruthlessly.
- **Read it aloud.** Your ear catches problems your eye skips — awkward phrasing, missing words, sentences that run too long. If you stumble reading it, readers will too.
- **Cut 20%.** After your first draft, challenge yourself to cut 20% of the word count without losing meaning. You almost always can.
- **Kill your darlings.** That clever phrase or perfect analogy? If it doesn't serve the reader, cut it. Writing is for the reader, not the writer.
- **Let it rest.** If possible, wait a day before final edits. Fresh eyes find what tired eyes miss.
- **Have someone else read it.** You can't objectively evaluate your own writing. Another person will find the gaps in logic, the unclear references, and the assumptions you didn't realize you were making.

## Common Mistakes

- **"It's" is "it is." "Its" is possessive.** The most common error in English. Always expand the contraction to check.
- **"Their/there/they're" and "your/you're."** Read the sentence with the expanded form. "You are welcome" works. "You are code" doesn't.
- **Dangling modifiers.** "Running down the street, the building came into view." The building was running? Fix: "Running down the street, I saw the building come into view."
- **Comma splices.** Two independent clauses need a period, semicolon, or conjunction — not just a comma. "It was raining, we stayed inside" should be "It was raining, so we stayed inside."
- **Inconsistent lists.** If bullet one starts with a verb, they all should. If bullet one is a sentence, they all should be. Parallel structure makes lists scannable.

## Writing Checklist

Before publishing, verify:

1. Can a reader understand the main point from the first paragraph?
2. Is every sentence necessary? Could any be cut without losing meaning?
3. Are there any jargon terms that aren't defined?
4. Is the tone consistent throughout?
5. Have you read it aloud?
6. Does the structure make sense when you read only the headings?
7. Are all claims supported with specifics, not vague assertions?
8. Has someone other than you reviewed it?
