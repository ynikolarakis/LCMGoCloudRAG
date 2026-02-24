"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { ConversationList } from "@/components/chat/ConversationList";
import { Separator } from "@/components/ui/separator";
import type { ConversationResponse } from "@/lib/api";

export default function ChatPage() {
  const t = useTranslations("chat");
  const queryClient = useQueryClient();
  const [conversationId, setConversationId] = useState<string | null>(null);

  const handleConversationSelect = useCallback(
    (conv: ConversationResponse | null) => {
      setConversationId(conv?.id ?? null);
    },
    [],
  );

  const handleConversationChange = useCallback(
    (convId: string) => {
      setConversationId(convId);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    [queryClient],
  );

  return (
    <div className="flex h-screen" data-testid="chat-page">
      {/* Conversation sidebar - visible on md+ screens */}
      <aside
        className="hidden w-64 shrink-0 flex-col border-r md:flex"
        data-testid="conversation-sidebar"
      >
        <div className="flex h-16 items-center px-4">
          <p className="text-sm font-medium text-muted-foreground">
            {t("conversations")}
          </p>
        </div>
        <Separator />
        <div className="flex-1 overflow-y-auto p-3">
          <ConversationList
            activeConversationId={conversationId}
            onSelect={handleConversationSelect}
          />
        </div>
      </aside>

      {/* Chat area */}
      <div className="flex flex-1 flex-col" data-testid="chat-area">
        <ChatInterface
          conversationId={conversationId}
          onConversationChange={handleConversationChange}
        />
      </div>
    </div>
  );
}
