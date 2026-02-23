export type ConnectionStatus = "connected" | "reconnecting" | "disconnected";

export interface WsCitation {
  source: string;
  page: number | null;
  content_preview: string;
}

export interface WebSocketCallbacks {
  onToken?: (content: string) => void;
  onCitations?: (citations: WsCitation[]) => void;
  onDone?: (latencyMs: number) => void;
  onError?: (detail: string) => void;
  onStatusChange?: (status: ConnectionStatus) => void;
}

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/ws";
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private callbacks: WebSocketCallbacks = {};
  private retryCount = 0;
  private messageQueue: string[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private status: ConnectionStatus = "disconnected";
  private _token: string | null = null;

  connect(callbacks: WebSocketCallbacks, token?: string | null): void {
    this.callbacks = callbacks;
    this._token = token || null;
    this.retryCount = 0;
    this._connect();
  }

  private _connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const url = this._token
      ? `${WS_BASE}?token=${this._token}`
      : WS_BASE;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.retryCount = 0;
      this._setStatus("connected");

      while (this.messageQueue.length > 0) {
        const msg = this.messageQueue.shift()!;
        this.ws?.send(msg);
      }
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as Record<
          string,
          unknown
        >;
        this._handleMessage(data);
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  private _handleMessage(data: Record<string, unknown>): void {
    switch (data.type) {
      case "ping":
        this.ws?.send(JSON.stringify({ type: "pong" }));
        break;
      case "token":
        this.callbacks.onToken?.(data.content as string);
        break;
      case "citations":
        this.callbacks.onCitations?.(data.citations as WsCitation[]);
        break;
      case "done":
        this.callbacks.onDone?.(data.latency_ms as number);
        break;
      case "error":
        this.callbacks.onError?.(data.detail as string);
        break;
      case "status":
        // processing status — no-op
        break;
    }
  }

  private _scheduleReconnect(): void {
    if (this.retryCount >= MAX_RETRIES) {
      this._setStatus("disconnected");
      return;
    }

    this._setStatus("reconnecting");
    const delay = BASE_DELAY_MS * Math.pow(2, this.retryCount);
    this.retryCount++;

    this.reconnectTimer = setTimeout(() => {
      this._connect();
    }, delay);
  }

  private _setStatus(status: ConnectionStatus): void {
    this.status = status;
    this.callbacks.onStatusChange?.(status);
  }

  send(message: {
    type: string;
    question?: string;
    conversation_id?: string;
  }): void {
    const json = JSON.stringify(message);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(json);
    } else {
      this.messageQueue.push(json);
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this._setStatus("disconnected");
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }
}
