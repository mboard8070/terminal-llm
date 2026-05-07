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
- [x] Parallel tool execution (read-only tools run concurrently; mutating tools stay sequential)
- [x] Pytest suite organized around a testing pyramid: unit coverage, integration coverage, and gateway HTTP smoke tests
- [x] Phone location awareness
- [x] Long-term memory / context retention across sessions
- [x] PDF and Office document parsing tools

## Open

### Phone App
- [ ] No open items

### Quality of Life
- [ ] No open items

### Infrastructure
- [ ] Gateway connection pooling for cloud API requests
- [ ] Extend transient retry/error recovery to plain provider proxy and frontier calls
