"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConnectionIndicator } from "./ConnectionIndicator";
import { MessageList } from "./MessageList";
import type { ChatMessage } from "./MessageBubble";
import {
  WebSocketManager,
  type ConnectionStatus,
  type WsCitation,
} from "@/lib/websocket";
import { useAuth } from "@/components/AuthProvider";

export function ChatInterface() {
  const t = useTranslations("chat");
  const { token } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocketManager | null>(null);

  useEffect(() => {
    const ws = new WebSocketManager();
    wsRef.current = ws;

    ws.connect({
      onToken: (content: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant" && last.isStreaming) {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + content,
            };
          }
          return updated;
        });
      },
      onCitations: (citations: WsCitation[]) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, citations };
          }
          return updated;
        });
      },
      onDone: () => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, isStreaming: false };
          }
          return updated;
        });
        setIsStreaming(false);
      },
      onError: (detail: string) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: detail, isStreaming: false },
        ]);
        setIsStreaming(false);
      },
      onStatusChange: setConnectionStatus,
    }, token);

    return () => {
      ws.disconnect();
    };
  }, [token]);

  const handleSend = useCallback(() => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", isStreaming: true },
    ]);
    setInput("");
    setIsStreaming(true);

    wsRef.current?.send({ type: "query", question });
  }, [input, isStreaming]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen flex-col" data-testid="chat-interface">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{t("title")}</h2>
        <ConnectionIndicator status={connectionStatus} />
      </div>

      {/* Messages */}
      <MessageList messages={messages} />

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t("placeholder")}
            disabled={isStreaming}
            data-testid="chat-input"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            data-testid="chat-send-button"
            aria-label={t("sendButton")}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
