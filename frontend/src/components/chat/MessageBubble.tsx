"use client";

import { useMemo } from "react";
import DOMPurify from "dompurify";
import { cn } from "@/lib/utils";
import { CitationCard } from "./CitationCard";
import type { WsCitation } from "@/lib/websocket";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: WsCitation[];
  isStreaming?: boolean;
}

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  const sanitizedContent = useMemo(() => {
    if (isUser) return message.content;
    return DOMPurify.sanitize(message.content);
  }, [message.content, isUser]);

  return (
    <div
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
      data-testid={`message-${message.role}`}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-3",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted",
        )}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{sanitizedContent}</p>
        ) : (
          <div
            className="text-sm whitespace-pre-wrap prose prose-sm dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{ __html: sanitizedContent }}
          />
        )}

        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-foreground/50 animate-pulse" />
        )}

        {message.citations && message.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1" data-testid="citations-list">
            {message.citations.map((citation, i) => (
              <CitationCard key={i} citation={citation} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
