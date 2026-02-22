import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useChat } from "./hooks/useChat";
import { useConversations } from "./hooks/useConversations";
import { saveMessages } from "./hooks/storage";
import { MessageBubble } from "./components/MessageBubble";
import { ChatInput } from "./components/ChatInput";
import { ModelSelector } from "./components/ModelSelector";
import { ConversationDrawer } from "./components/ConversationDrawer";

/** Inner view — remounted via `key` when activeId changes */
const ChatView: FC<{
  conversationId: string | null;
  onFirstMessage: (text: string, model: string) => string;
  onMessageSent: () => void;
  onOpenDrawer: () => void;
  onNewChat: () => void;
}> = ({ conversationId, onFirstMessage, onMessageSent, onOpenDrawer, onNewChat }) => {
  const navigate = useNavigate();
  const scrollRef = useRef<HTMLDivElement>(null);
  const convIdRef = useRef(conversationId);
  const { messages, isStreaming, currentModel, setCurrentModel, autoRoute, setAutoRoute, sendMessage, stopStreaming } = useChat(conversationId);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = useCallback(
    (text: string, imageUrl?: string) => {
      if (!convIdRef.current) {
        const displayText = text || (imageUrl ? "Image conversation" : "New chat");
        const newId = onFirstMessage(displayText, currentModel);
        convIdRef.current = newId;
      }
      sendMessage(text, imageUrl);
      onMessageSent();
    },
    [sendMessage, onFirstMessage, onMessageSent, currentModel],
  );

  // Persist messages whenever they change
  useEffect(() => {
    if (convIdRef.current && messages.length > 0) {
      saveMessages(convIdRef.current, messages);
    }
  }, [messages]);

  return (
    <>
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-maude-border bg-maude-surface px-4 py-2">
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenDrawer}
            className="rounded-lg bg-maude-bg px-2 py-1 text-sm text-maude-muted hover:text-maude-text"
            aria-label="Open conversations"
          >
            &#9776;
          </button>
          <h1 className="fire-gradient text-lg font-bold">MAUDE</h1>
          <button
            onClick={onNewChat}
            className="rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-muted hover:text-maude-text"
          >
            New
          </button>
          <button onClick={() => navigate("/maude/voice")} className="rounded-lg bg-maude-bg px-2 py-1 text-xs text-maude-accent hover:text-maude-text">{"\uD83C\uDF99\uFE0F"} Voice</button>
        </div>
        <ModelSelector currentModel={currentModel} onSelect={setCurrentModel} autoRoute={autoRoute} onToggleAutoRoute={setAutoRoute} />
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="no-scrollbar flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <span className="fire-gradient mb-3 text-5xl font-black">{"\u25C7"}</span>
            <h2 className="mb-1 text-lg font-semibold text-maude-text">MAUDE</h2>
            <p className="max-w-xs text-sm text-maude-muted">Your local AI assistant. Powered by Mistral &amp; Codestral. Ask me anything.</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {["What can you do?", "Write a Python script", "Explain this code", "System status"].map((q) => (
                <button key={q} onClick={() => handleSend(q)}
                  className="rounded-full border border-maude-border px-3 py-1.5 text-xs text-maude-muted transition-colors hover:border-maude-accent hover:text-maude-text">{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg) => (<MessageBubble key={msg.id} message={msg} />))}
      </div>

      <ChatInput
        onSend={(text, imageUrl) => handleSend(text, imageUrl)}
        isStreaming={isStreaming}
        onStop={stopStreaming}
      />
    </>
  );
};

export const Chat: FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const {
    conversations,
    activeId,
    createConversation,
    switchConversation,
    deleteConversation,
    touchConversation,
    startNewChat,
  } = useConversations();

  const handleFirstMessage = useCallback(
    (text: string, model: string): string => {
      return createConversation(text, model);
    },
    [createConversation],
  );

  const handleMessageSent = useCallback(() => {
    if (activeId) touchConversation(activeId);
  }, [activeId, touchConversation]);

  return (
    <div className="flex h-full flex-col">
      <ChatView
        key={activeId || "new"}
        conversationId={activeId}
        onFirstMessage={handleFirstMessage}
        onMessageSent={handleMessageSent}
        onOpenDrawer={() => setDrawerOpen(true)}
        onNewChat={startNewChat}
      />

      <ConversationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        conversations={conversations}
        activeId={activeId}
        onSelect={switchConversation}
        onDelete={deleteConversation}
        onNewChat={startNewChat}
      />
    </div>
  );
};
