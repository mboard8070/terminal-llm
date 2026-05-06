# Prompt Engineering Best Practices

## Core Principles

- **Be specific, not vague.** "Summarize this in 3 bullet points, each under 20 words" beats "Summarize this." Constraints produce better output than open-ended requests.
- **Show, don't just tell.** Include examples of the desired output format. One good example is worth a paragraph of instructions.
- **Put the most important instructions first and last.** Models pay more attention to the beginning and end of prompts. Bury critical rules in the middle and they'll be missed.
- **One task per prompt when possible.** Asking a model to analyze, summarize, translate, and format in one prompt degrades quality on each sub-task. Chain separate calls for complex workflows.
- **Iterate, don't overthink.** Start with a simple prompt, see what's wrong with the output, then add constraints to fix it. Prompt engineering is debugging, not architecture.

## System Prompts

- **Define the role clearly.** "You are a senior Python developer reviewing code for production readiness" gives the model a persona that shapes all subsequent responses.
- **State what NOT to do.** Models follow positive instructions better, but explicit prohibitions prevent common failure modes. "Do NOT include explanations — return only the code."
- **Set the output format upfront.** "Always respond in JSON with keys: summary, confidence, sources" prevents format drift across a conversation.
- **Keep system prompts under 2000 tokens for simple tasks.** Longer prompts dilute focus. If your system prompt is 5 pages, the model will forget parts of it.
- **For complex agents, structure with headers.** Use markdown sections: `## Role`, `## Rules`, `## Tools`, `## Output Format`. Models parse structured prompts better than walls of text.

## Few-Shot Examples

- **2-3 examples is the sweet spot.** One example might be coincidence. Five is usually redundant. Show the pattern with 2-3 diverse cases.
- **Include edge cases in examples.** If you want the model to handle empty input or ambiguous cases, show it one.
- **Format examples identically.** Use consistent delimiters (`---`, `###`, XML tags) between examples. The model learns the pattern.
- **Example format:**
  ```
  Input: "The product arrived broken and customer service was unhelpful"
  Output: {"sentiment": "negative", "topics": ["product_quality", "customer_service"]}

  Input: "Fast shipping, exactly what I ordered"
  Output: {"sentiment": "positive", "topics": ["shipping", "accuracy"]}
  ```

## Structured Output

- **Request JSON explicitly.** "Respond with valid JSON only. No markdown, no explanation, no code fences." Models default to conversational responses.
- **Provide the JSON schema.** Show the exact structure with field names, types, and example values. Don't let the model invent field names.
- **Use XML tags for complex prompts.** `<context>`, `<instructions>`, `<output>` tags help models separate sections. Works well with Claude especially.
- **Specify arrays vs. objects.** "Return a JSON array of objects" vs. "Return a JSON object with a key 'items' containing an array" — the difference matters for parsing.

## Chain of Thought

- **Use "Think step by step" when reasoning matters.** Math, logic, code analysis, multi-step problems. The model's first instinct is often wrong; walking through steps catches errors.
- **Ask for the answer last.** "First analyze the code for bugs. Then explain each bug. Finally, provide the corrected code." Putting the final answer at the end gives the model room to reason.
- **Separate reasoning from output.** "Think through this in a <thinking> block, then provide your final answer in an <answer> block." This lets you parse the answer while keeping the reasoning for debugging.
- **Don't force chain-of-thought on simple tasks.** "What is the capital of France? Think step by step." — unnecessary overhead. Reserve it for tasks where reasoning improves accuracy.

## Context Management

- **Put reference material before instructions.** "Here is the codebase: [code]. Now, find all SQL injection vulnerabilities." The model needs context loaded before it knows what to do with it.
- **Label your context.** "The following is a customer support transcript:" is better than just pasting the transcript. The model needs to know what it's looking at.
- **Trim irrelevant context.** More context isn't always better. Irrelevant information can confuse the model or push important details out of the attention window.
- **Chunk large documents.** If a document exceeds the context window, summarize sections or process chunks independently, then combine results.

## Tool Use / Function Calling

- **Write tool descriptions like API docs.** Clear parameter descriptions, expected types, what the tool returns, when to use it vs. alternatives.
- **Include "when NOT to use" in tool descriptions.** "Use web_search for current events and real-time data. Do NOT use for general knowledge questions." Reduces unnecessary tool calls.
- **Minimal required parameters.** Every required parameter is a chance for the model to hallucinate a value. Make parameters optional with sensible defaults where possible.
- **Return structured, parseable results.** Tool results should be clean text or JSON. Don't return raw HTML or binary data and expect the model to parse it.
- **Name tools clearly.** `search_emails` is better than `email_tool`. `create_calendar_event` is better than `calendar`. The name should describe the action.

## Delegation / Sub-Agent Prompts

- **Be explicit about what to return.** "Return only the final result, not your reasoning or intermediate steps." Sub-agents that narrate their process waste tokens.
- **Set scope boundaries.** "Only answer questions about Python. For other languages, respond with 'I can only help with Python.'" Prevents scope creep in specialized agents.
- **Include the original user intent.** When delegating, pass the user's actual question plus context, not just a reformulated version. The sub-agent might interpret differently.
- **Specify the output contract.** "Return a JSON object with 'answer' (string) and 'confidence' (float 0-1)." Sub-agent outputs need to be parseable by the calling agent.

## Image Generation Prompts

- **Front-load the subject.** "A golden retriever sitting in a sunlit meadow" not "In a meadow where the sun is shining, there can be seen a dog of the golden retriever breed."
- **Style before details.** "Oil painting style, impressionist. A woman reading in a cafe..." The style modifier shapes everything that follows.
- **Be specific about composition.** "Close-up portrait", "wide-angle landscape", "bird's-eye view", "centered subject with shallow depth of field." Don't leave framing to chance.
- **Negative prompts matter.** Specify what to avoid: "no text, no watermarks, no extra fingers, no distorted faces."
- **Quality boosters.** "High quality, detailed, professional photography, 8K, sharp focus" — these modifiers consistently improve output across models.
- **Aspect ratio affects composition.** Specify it. Portraits: 2:3 or 3:4. Landscapes: 16:9 or 3:2. Square: 1:1 for social media.

## Common Anti-Patterns

- **Don't be polite to the model.** "Could you please kindly summarize..." wastes tokens. "Summarize:" works identically.
- **Don't repeat instructions.** Saying the same rule 3 different ways doesn't make it stronger. Say it once, clearly.
- **Don't use ambiguous references.** "Do the same thing as before" — the model may not interpret "before" the way you expect. Be explicit.
- **Don't assume the model remembers.** In long conversations, re-state critical constraints. Context windows are large but attention is uneven.
- **Don't ask "Do you understand?"** The model will always say yes. Instead, ask it to restate the task in its own words if you want to verify comprehension.

## Prompt Engineering Checklist

Before finalizing a prompt, verify:

1. Is the task clearly defined in the first sentence?
2. Is the output format specified explicitly?
3. Are there examples of expected input/output?
4. Are edge cases addressed?
5. Is irrelevant context removed?
6. Are constraints specific and measurable (not "be concise" but "under 100 words")?
7. Have you tested with adversarial inputs?
8. Is the prompt as short as it can be while remaining unambiguous?
