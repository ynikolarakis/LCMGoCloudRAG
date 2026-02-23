"use client";

import { useState } from "react";
import { FileText } from "lucide-react";
import type { WsCitation } from "@/lib/websocket";
import { Badge } from "@/components/ui/badge";

interface CitationCardProps {
  citation: WsCitation;
  index: number;
}

export function CitationCard({ citation, index }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <button
      className="inline-flex items-center gap-1 rounded-md border bg-muted px-2 py-1 text-xs hover:bg-accent transition-colors text-left"
      onClick={() => setExpanded(!expanded)}
      data-testid={`citation-${index}`}
    >
      <FileText className="h-3 w-3 shrink-0" />
      <span className="font-medium">{citation.source}</span>
      {citation.page != null && (
        <Badge variant="secondary" className="text-xs px-1 py-0">
          p.{citation.page}
        </Badge>
      )}
      {expanded && citation.content_preview && (
        <span className="block mt-1 text-muted-foreground whitespace-pre-wrap">
          {citation.content_preview}
        </span>
      )}
    </button>
  );
}
