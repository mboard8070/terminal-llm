# MAUDE TODO

## Done
- [x] Multi-agent architecture (subagents: code, vision, writer, reasoning, research, search)
- [x] Cross-machine task execution (collab system, client task executor)
- [x] Auto-routing (intent detection, model switching)
- [x] Cloud integration (Claude, Mistral, Codestral, Devstral, Grok, Gemini)
- [x] Local model inference (Nemotron on llama-server)
- [x] Streaming responses with typewriter effect
- [x] Google Workspace integration (43 tools: Gmail, Drive, Sheets, Slides, Calendar, Contacts, YouTube)
- [x] Voice interaction (Nemotron ASR + Magpie TTS, full tool access)
- [x] Mobile app (iOS, Capacitor, EventSource streaming)
- [x] Tool system (fast dispatch, keyword-based tool groups, scoped tool access)
- [x] Image analysis (LLaVA)
- [x] Web search and browsing
- [x] Gateway with HTTPS + HTTP mirror
- [x] OAuth app verification for persistent Google tokens

## Open

### Phone App
- [ ] Phone location awareness (send GPS coords as context)
- [ ] Push notifications
- [ ] Nemotron reasoning_content display (fix merged, needs deploy to TestFlight)

### Local Model
- [ ] Nemotron tool calling (gated — only when query clearly needs tools, otherwise route to cloud)

### Quality of Life
- [ ] Long-term memory / context retention across sessions
- [ ] Slack/Discord integration
- [ ] PDF and Office document parsing tools
- [ ] Parallel tool execution (run independent tool calls concurrently)

### Infrastructure
- [ ] Gateway connection pooling for cloud API requests
- [ ] Better error recovery and auto-retry for transient failures
