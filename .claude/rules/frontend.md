---
globs: frontend/**
---

# Frontend Rules

## Tech Stack
- Next.js 15 (App Router with Server Components)
- TypeScript strict mode — no `any` types, no `@ts-ignore`
- Tailwind CSS + shadcn/ui (never write raw CSS unless absolutely necessary)
- React Query (TanStack Query v5) for all server state management
- next-intl for i18n (Greek `el` + English `en`)
- Keycloak JS adapter for SSO authentication
- Lucide React for icons

## Component Standards
- Functional components with hooks ONLY (no class components)
- Named exports for all components (no `export default`)
- Props: TypeScript interface, destructured in function signature
- Naming: PascalCase for components, camelCase for hooks/utils
- One component per file, filename matches component name

## State Management
- Server state: React Query (queries, mutations, optimistic updates)
- Local UI state: useState / useReducer
- Form state: React Hook Form + Zod validation
- Global state: React Context only for auth/locale (no Redux)
- NEVER duplicate server state in local state

## API Client
- All API calls go through typed functions in `lib/api/`
- Never use raw `fetch()` in components
- Base URL from environment variable
- Auth token injected automatically via interceptor
- Response types match Pydantic models from backend

## Security
- JWT stored in httpOnly secure cookies (NEVER localStorage or sessionStorage)
- CSRF token on all mutation requests (POST, PUT, DELETE)
- Sanitize user-generated content with DOMPurify before rendering
- No `dangerouslySetInnerHTML` without DOMPurify
- Content-Security-Policy headers in `next.config.js`
- No inline scripts or styles in production

## i18n
- next-intl with locale prefix routing (`/en/chat`, `/el/chat`)
- Message files: `messages/en.json`, `messages/el.json`
- Namespace-based: `chat.sendButton`, `documents.uploadTitle`, `admin.healthStatus`
- Dates and numbers: use `Intl.DateTimeFormat` and `Intl.NumberFormat`
- Default locale: English

## WebSocket (Chat)
- Single WebSocket manager in `lib/websocket.ts`
- Auto-reconnect with exponential backoff (max 5 retries)
- Heartbeat: ping every 30 seconds
- Stream tokens into UI as they arrive
- Buffer complete message for citation extraction after stream ends
- Connection state indicator in chat UI

## Accessibility (WCAG 2.1 AA)
- Semantic HTML: proper heading hierarchy (h1 → h2 → h3)
- All interactive elements keyboard-accessible
- `aria-label` on icon-only buttons
- Focus management in modals and dialogs (trap focus, restore on close)
- Screen reader announcements for dynamic content (`aria-live="polite"`)
- Color contrast: minimum 4.5:1 for text

## Testing
- Vitest for unit tests
- React Testing Library for component tests
- Playwright for E2E tests (via Playwright MCP server)
- Test IDs: `data-testid="chat-send-button"` on all interactive elements
