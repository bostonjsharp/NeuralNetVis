// Local stand-in for the footron messaging router, so phone controls can be
// exercised without the wall. Two endpoints on one port:
//
//   ws://localhost:8089/out    the wall app connects here (pass it as ?ftMsgUrl,
//                              or just use ?ftmsg=1 — the vendored client's
//                              default URL is exactly this one)
//   http://localhost:8089/     a phone-simulator page with the same draw pad
//                              the real controls panel has; it relays over /phone
//
// The real router wraps every phone message in an envelope before the wall
// sees it. This mock speaks just enough of that protocol for the app side:
//   router → wall:  { type: "con", client: id }             phone joined
//                   { type: "cap", client: id, body: ... }  phone message
// Everything the wall sends back (accepts, heartbeats) is logged and ignored.
//
// Usage:  npm run mock     then open the app with ?ftmsg=1
import { createServer } from 'node:http'
import { WebSocketServer } from 'ws'

const PORT = 8089
const CLIENT_ID = 'dev-phone'

const walls = new Set()
let phoneJoined = false

const sendToWalls = (message) => {
  const data = JSON.stringify(message)
  for (const ws of walls) if (ws.readyState === ws.OPEN) ws.send(data)
}

// Kept visually in lockstep with controls/lib/index.js — the real panel only
// runs inside footron-web, so this page doubles as its local design preview.
const PAGE = /* html */ `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Net phone sim</title>
<style>
  body { background: #020308; font-family: system-ui, sans-serif;
         max-width: 420px; margin: 0 auto; padding: 12px; }
  .panel { display: flex; flex-direction: column; gap: 14px;
           padding: 18px 14px 26px; border-radius: 14px;
           background: radial-gradient(120% 90% at 50% 0%, #101733 0%, #050510 70%); }
  @keyframes nn-breathe { 0%, 100% { opacity: 0.35; } 50% { opacity: 0.9; } }
  .masthead { text-align: center; padding: 4px 0 2px; }
  .title { font-weight: 800; font-size: 22px; letter-spacing: 0.14em; margin-right: -0.14em;
           color: whitesmoke;
           text-shadow: 0 0 18px rgba(77,191,255,0.65), 0 0 40px rgba(77,191,255,0.3); }
  .sub { margin-top: 4px; font-size: 12px; letter-spacing: 0.22em; margin-right: -0.22em;
         color: #8d97b4; }
  .live { display: inline-block; width: 7px; height: 7px; margin: 0 8px 1px 0;
          border-radius: 50%; background: #ffa03d; box-shadow: 0 0 8px #ffa03d;
          animation: nn-breathe 2.2s ease-in-out infinite; }
  .pad-wrap { position: relative; }
  canvas { display: block; width: 100%; aspect-ratio: 1; border: 1px solid #273258;
           border-radius: 14px; background: #000; touch-action: none;
           box-shadow: inset 0 0 30px rgba(77,191,255,0.08), 0 0 22px rgba(77,191,255,0.12); }
  .pad-hint { position: absolute; inset: 0; display: grid; place-items: center;
              pointer-events: none; color: #8d97b4; font-size: 17px;
              letter-spacing: 0.12em; }
  .row { display: flex; gap: 14px; }
  button { border: 1px solid #273258; border-radius: 14px; font-weight: bolder;
           color: whitesmoke; background: #0d1226; cursor: pointer;
           transition: transform 90ms ease, box-shadow 90ms ease; }
  button:active { transform: scale(0.96); }
  .clear { flex: 1; height: 64px; font-size: 17px; letter-spacing: 0.16em; color: #4dbfff;
           border-color: rgba(77,191,255,0.45); }
  .clear:active { box-shadow: 0 0 22px rgba(77,191,255,0.35); }
  .brain { flex: 1; height: 64px; font-size: 17px; letter-spacing: 0.14em; color: #050510;
           background: linear-gradient(180deg, #ffc57a 0%, #ffa03d 55%, #e0862e 100%);
           border-color: #ffa03d; box-shadow: 0 0 24px rgba(255,160,61,0.35); }
  .brain:active { box-shadow: 0 0 40px rgba(255,160,61,0.6); }
  .foot { text-align: center; font-size: 12px; letter-spacing: 0.14em; color: #8d97b4;
          opacity: 0.8; }
</style>
<div class="panel">
  <div class="masthead">
    <div class="title">INSIDE A NEURAL NETWORK</div>
    <div class="sub"><span class="live"></span>DRAW-PAD LINK</div>
  </div>
  <div class="pad-wrap">
    <canvas id="pad"></canvas>
    <div class="pad-hint" id="hint">finger-draw a digit here</div>
  </div>
  <div class="row">
    <button class="clear" id="clear">clear</button>
    <button class="brain" id="brain">switch brain</button>
  </div>
  <div class="foot" id="status">connecting…</div>
</div>
<script>
  const status = document.getElementById('status')
  const ws = new WebSocket('ws://' + location.host + '/phone')
  ws.onopen = () => { status.textContent = 'connected — wall should wake' }
  ws.onclose = () => { status.textContent = 'disconnected (is the mock still running?)' }
  const send = (body) => { if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(body)) }

  const canvas = document.getElementById('pad')
  const hint = document.getElementById('hint')
  const size = Math.round(canvas.clientWidth * (window.devicePixelRatio || 1))
  canvas.width = size; canvas.height = size
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, size, size)

  let drawing = false, last = null, lastSent = 0, pending = null
  const clamp = (v) => v < 0 ? 0 : v > 1 ? 1 : v
  const toNorm = (e) => {
    const r = canvas.getBoundingClientRect()
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) }
  }
  const drawLocal = (from, to) => {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = size * 0.07
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'
    ctx.beginPath(); ctx.moveTo(from.x * size, from.y * size)
    ctx.lineTo(to.x * size, to.y * size); ctx.stroke()
  }
  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault(); canvas.setPointerCapture(e.pointerId)
    const p = toNorm(e)
    drawing = true; last = p; pending = null; lastSent = performance.now()
    hint.style.display = 'none'
    drawLocal(p, p)
    send({ type: 'pen', action: 'down', x: p.x, y: p.y })
  })
  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return
    const p = toNorm(e)
    drawLocal(last, p); last = p
    const now = performance.now()
    if (now - lastSent >= 33) { lastSent = now; pending = null; send({ type: 'pen', action: 'move', x: p.x, y: p.y }) }
    else pending = p
  })
  const up = () => {
    if (!drawing) return
    drawing = false
    if (pending) { send({ type: 'pen', action: 'move', x: pending.x, y: pending.y }); pending = null }
    send({ type: 'pen', action: 'up' })
    status.textContent = 'stroke sent — the wall auto-fires after a pause'
  }
  canvas.addEventListener('pointerup', up)
  canvas.addEventListener('pointercancel', up)
  document.getElementById('clear').addEventListener('click', () => {
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, size, size)
    hint.style.display = ''
    status.textContent = 'pad cleared'
    send({ type: 'clear' })
  })
  document.getElementById('brain').addEventListener('click', () => {
    status.textContent = 'switch brain sent'
    send({ type: 'brain' })
  })
</script>
`

const http = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
  res.end(PAGE)
})

const wss = new WebSocketServer({ server: http })

wss.on('connection', (ws, req) => {
  if (req.url.startsWith('/phone')) {
    console.log('[mock] phone sim connected')
    phoneJoined = true
    sendToWalls({ type: 'con', client: CLIENT_ID })
    ws.on('message', (data) => {
      let body
      try { body = JSON.parse(data) } catch { return }
      if (body.type !== 'pen' || body.action !== 'move') console.log('[mock] phone →', body)
      sendToWalls({ type: 'cap', client: CLIENT_ID, body })
    })
    return
  }
  console.log('[mock] wall app connected on', req.url)
  walls.add(ws)
  if (phoneJoined) sendToWalls({ type: 'con', client: CLIENT_ID })
  ws.on('message', (data) => console.log('[mock] wall →', String(data)))
  ws.on('close', () => walls.delete(ws))
})

http.listen(PORT, () =>
  console.log(
    `[mock] router on ws://localhost:${PORT}/out — phone sim on http://localhost:${PORT}/`
  )
)
