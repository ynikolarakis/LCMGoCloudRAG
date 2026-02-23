"use client";

import { useEffect, useRef } from "react";
import { MessageBubble, type ChatMessage } from "./MessageBubble";
import { ScrollArea } from "@/components/ui/scroll-area";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <ScrollArea className="flex-1 p-4" data-testid="message-list">
      <div className="space-y-4">
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
