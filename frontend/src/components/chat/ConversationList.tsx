"use client";

import { useTranslations } from "next-intl";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  fetchConversations,
  deleteConversation,
  type ConversationResponse,
} from "@/lib/api";

interface ConversationListProps {
  activeConversationId: string | null;
  onSelect: (conv: ConversationResponse | null) => void;
}

export function ConversationList({
  activeConversationId,
  onSelect,
}: ConversationListProps) {
  const t = useTranslations("chat");
  const queryClient = useQueryClient();

  const { data } = useQuery({
    queryKey: ["conversations"],
    queryFn: () => fetchConversations(1, 50),
    refetchInterval: 10000,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteConversation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      onSelect(null);
    },
  });

  const conversations = data?.items ?? [];

  return (
    <div className="flex flex-col gap-1" data-testid="conversation-list">
      <Button
        variant="outline"
        size="sm"
        className="mb-2 w-full justify-start gap-2"
        onClick={() => onSelect(null)}
        data-testid="new-conversation-button"
      >
        <MessageSquarePlus className="h-4 w-4" />
        {t("newConversation")}
      </Button>

      {conversations.length === 0 && (
        <p className="px-2 text-xs text-muted-foreground">{t("noConversations")}</p>
      )}

      {conversations.map((conv) => (
        <div
          key={conv.id}
          className={cn(
            "group flex items-center justify-between rounded-md px-2 py-1.5 text-sm cursor-pointer hover:bg-accent",
            activeConversationId === conv.id && "bg-accent",
          )}
          onClick={() => onSelect(conv)}
          data-testid={`conversation-${conv.id}`}
        >
          <span className="truncate">{conv.title || t("newConversation")}</span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 opacity-0 group-hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation();
              deleteMutation.mutate(conv.id);
            }}
            aria-label={t("deleteConversation")}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}
