/** Phone controls for the footron wall.
 *
 *  The wall's physical controller plugs into footron's Windows machine, and web
 *  experiences run in a browser on the Linux machine — no input path connects
 *  them. Footron's own answer is the visitor's phone: it serves our
 *  controls/lib/index.js panel to the phone, and the panel talks to this page
 *  through the wall's messaging router over a WebSocket.
 *
 *  Protocol (keep in sync with controls/lib/index.js):
 *
 *    { type: "pen", action: "down" | "move", x: 0..1, y: 0..1 }
 *    { type: "pen", action: "up" }
 *    { type: "clear" }
 *    { type: "brain" }
 *
 *  The pen stream drives the exact same DrawPadHandle path the mouse does, so
 *  the app never knows which surface a stroke came from. Anything
 *  unrecognised is ignored: a panel newer than the deployed wall build should
 *  degrade rather than throw. */

export type PhoneCommand =
  | { type: "pen"; action: "down" | "move"; x: number; y: number }
  | { type: "pen"; action: "up" }
  | { type: "clear" }
  | { type: "brain" };

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Footron passes the router URL as `?ftMsgUrl=…`. Without it the vendored
 *  client retries ws://localhost:8089 forever, so off the wall we simply don't
 *  connect — `?ftmsg=1` forces it on for local testing against the dev mock. */
export function phoneEnabled(search?: string): boolean {
  const q = search ?? (typeof location === "undefined" ? "" : location.search);
  const params = new URLSearchParams(q);
  return params.has("ftMsgUrl") || params.get("ftmsg") === "1";
}

/** Pure message → command mapping, so the whole protocol is unit-testable.
 *  Coordinates are clamped into the pad; a malformed number kills the
 *  message, never produces NaN strokes. */
export function parsePhoneMessage(body: unknown): PhoneCommand | null {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null;
  const msg = body as { type?: unknown; action?: unknown; x?: unknown; y?: unknown };
  if (msg.type === "clear") return { type: "clear" };
  if (msg.type === "brain") return { type: "brain" };
  if (msg.type !== "pen") return null;
  if (msg.action === "up") return { type: "pen", action: "up" };
  if (msg.action !== "down" && msg.action !== "move") return null;
  if (typeof msg.x !== "number" || !Number.isFinite(msg.x)) return null;
  if (typeof msg.y !== "number" || !Number.isFinite(msg.y)) return null;
  return { type: "pen", action: msg.action, x: clamp01(msg.x), y: clamp01(msg.y) };
}

type PhoneHandlers = {
  onCommand: (cmd: PhoneCommand) => void;
  /** A phone joined — scanning the QR code should feel acknowledged even
   *  before the first stroke. */
  onConnect?: () => void;
};

// the vendored UMD script defines this global; loading it is index.html's job
type MessagingLib = {
  Messaging: new () => {
    mount: () => void;
    unmount: () => void;
    addMessageListener: (cb: (body: unknown) => void) => void;
    removeMessageListener: (cb: (body: unknown) => void) => void;
    addConnectionListener: (cb: (connection: unknown) => void) => void;
    removeConnectionListener: (cb: (connection: unknown) => void) => void;
  };
};

/** Connect to the wall's messaging router and feed phone commands to the app.
 *  Returns a teardown function. Safe to call off the wall — it no-ops, and
 *  no-ops again if the vendored script failed to load, because a missing
 *  script must not take the exhibit down with it. */
export function connectPhone(handlers: PhoneHandlers, opts?: { enabled?: boolean }): () => void {
  const enabled = opts?.enabled ?? phoneEnabled();
  if (!enabled) return () => {};

  const lib = (globalThis as { FootronMessaging?: MessagingLib }).FootronMessaging;
  if (!lib || typeof lib.Messaging !== "function") {
    console.warn("[phone] footron messaging client not loaded; phone controls are off");
    return () => {};
  }

  const client = new lib.Messaging();
  const onMessage = (body: unknown) => {
    const cmd = parsePhoneMessage(body);
    if (cmd) handlers.onCommand(cmd);
  };
  const onConnection = () => handlers.onConnect?.();
  client.addMessageListener(onMessage);
  client.addConnectionListener(onConnection);
  client.mount();
  return () => {
    client.removeMessageListener(onMessage);
    client.removeConnectionListener(onConnection);
    client.unmount();
  };
}
