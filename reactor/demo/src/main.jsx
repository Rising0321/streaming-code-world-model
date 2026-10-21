import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReactorProvider, ReactorView, useReactor, useReactorMessage } from "@reactor-team/js-sdk";
import { Logo } from "@reactor-team/ui";
import { selectExample, sourceFile, stateReply, validateSelection } from "./inputs.js";
import "./style.css";

function usePreview(source) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!source) { setUrl(""); return; }
    const value = typeof source === "string" ? source : URL.createObjectURL(source);
    setUrl(value);
    return () => { if (typeof source !== "string") URL.revokeObjectURL(value); };
  }, [source]);
  return url;
}

function Workspace({ endpoint, onEndpointChange }) {
  const status = useReactor((s) => s.status);
  const connect = useReactor((s) => s.connect), disconnect = useReactor((s) => s.disconnect);
  const reconnect = useReactor((s) => s.reconnect);
  const send = useReactor((s) => s.sendCommand), upload = useReactor((s) => s.uploadFile);
  const lastError = useReactor((s) => s.lastError);
  const reactor = useReactor((s) => s.internal.reactor);
  const [examples, setExamples] = useState([]), [catalogError, setCatalogError] = useState("");
  const [selected, setSelected] = useState("");
  const [conditions, setConditions] = useState(null), [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null), [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0), [limit, setLimit] = useState(24);
  const [world, setWorld] = useState(null), [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [canReconnect, setCanReconnect] = useState(false);
  const epoch = useRef(0), inCommand = useRef(false);
  const referenceTrack = useReactor((s) => s.tracks.reference_video);
  const imageUrl = usePreview(image);
  const active = Boolean(world?.active), locked = active || Boolean(busy);
  useEffect(() => {
    if (lastError) {
      setError(String(lastError));
      if (lastError.code === "DISCONNECTED" && lastError.recoverable)
        setCanReconnect(true);
    }
  }, [lastError]);
  useReactorMessage((message) => { if (message.type === "state_update") setWorld(message.data); });
  useEffect(() => {
    if (status !== "ready") { epoch.current++; setWorld(null); }
  }, [status]);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/examples/index.json", { signal: controller.signal })
      .then((r) => { if (!r.ok) throw Error("Examples are not installed. Run prepare_examples.py; custom uploads remain available."); return r.json(); })
      .then(setExamples).catch((e) => { if (e.name !== "AbortError") setCatalogError(e.message); });
    return () => controller.abort();
  }, []);
  async function command(fn) {
    if (inCommand.current) return;
    inCommand.current = true; setError("");
    try { await fn(); } catch (e) { setError(String(e)); }
    finally { inCommand.current = false; setBusy(""); }
  }
  async function sendState(name, data) {
    const previous = reactor.getLastError();
    const reply = await send(name, data);
    const failure = reactor.getLastError();
    return stateReply(reply, failure !== previous ? failure : undefined);
  }
  async function toggleConnection() {
    await command(async () => {
      setBusy(status === "ready" ? "Disconnecting…" : canReconnect ? "Reconnecting to the existing render…" : "Connecting…");
      if (status === "ready") { setCanReconnect(false); await disconnect(); setWorld(null); }
      else {
        let address;
        try { address = new URL(endpoint.trim()); }
        catch { throw Error("Use an HTTP or HTTPS Runtime endpoint."); }
        if (!["http:", "https:"].includes(address.protocol)) throw Error("Use an HTTP or HTTPS Runtime endpoint.");
        if (canReconnect) { setCanReconnect(false); await reconnect(); }
        else await connect();
        if (reactor.getStatus() === "ready") setCanReconnect(false);
      }
    });
  }
  function chooseExample(name) {
    const item = selectExample(examples, name);
    setSelected(name); setError("");
    setConditions(item.conditions); setImage(item.image); setPreview(item.preview);
    setPrompt(item.prompt); setLimit(item.limit);
  }
  async function render() {
    await command(async () => {
      validateSelection({ conditions, image, prompt, seed, limit });
      if (status !== "ready") throw Error("Connect before rendering.");
      const token = epoch.current;
      const check = () => { if (token !== epoch.current) throw Error("Connection changed. Reconnect and start again."); };
      if (selected) {
        setBusy("Starting the selected server-side example…");
        const answer = await sendState("start_example", { example_id: Number(selected.slice(-2)), prompt, seed, max_chunks: limit });
        check();
        setWorld((current) => current?.world_id === answer.world_id ? current : answer);
        return;
      }
      setBusy("Loading selected input files…");
      const [c, i] = await Promise.all([sourceFile(conditions, "conditions"), sourceFile(image, "image")]);
      check(); setBusy(`Uploading lossless conditions (${(c.size / 1048576).toFixed(1)} MiB)…`);
      const conditionUpload = await upload(c);
      check(); setBusy("Uploading reference image…");
      const imageUpload = await upload(i);
      check(); setBusy("Validating inputs…");
      const answer = await sendState("start", { conditions: conditionUpload, image: imageUpload, prompt, seed, max_chunks: limit });
      check();
      setWorld((current) => current?.world_id === answer.world_id ? current : answer);
    });
  }
  async function stop() {
    await command(async () => {
      setBusy("Stopping at the next chunk boundary…");
      const answer = await sendState("stop", {}); setWorld(answer);
    });
  }
  const progress = busy || (world?.complete ? "Complete. Buffered video may still be playing." : active ?
    world.ready ? "Rendering the next native chunk…" : "Encoding inputs and preparing the first two chunks…" : "Choose inputs, then Start rendering.");
  const message = error;
  const filename = (x, fallback) => x ? (typeof x === "string" ? x.split("/").at(-1) : x.name) : fallback;
  return <div className="shell">
    <header>
      <a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer" aria-label="Reactor"><Logo variant="symbol" color="white" height={20} /></a>
      <i /><strong>Streaming Code World Model</strong><span className="tag">RECORDED CONDITIONS</span>
      <div className="connection">
        <label htmlFor="endpoint">Runtime</label><input id="endpoint" type="url" value={endpoint} spellCheck={false}
          placeholder="http://localhost:8080" disabled={status !== "disconnected" || Boolean(busy)}
          onChange={(e) => { onEndpointChange(e.target.value); setError(""); setCanReconnect(false); }} />
        <b className={`dot ${status === "ready" ? "online" : ""}`} /><span>{status === "ready" ? "Connected" : status}</span>
        <button onClick={toggleConnection} disabled={Boolean(busy) || status === "connecting" || status === "waiting"}>{status === "ready" ? "Disconnect" : canReconnect ? "Reconnect" : "Connect"}</button>
      </div>
    </header>
    <main className="workspace">
      <aside className="sidebar">
        <span className="eyebrow">01 / CHOOSE YOUR INPUTS</span><h1>A recorded world,<br />one chunk at a time.</h1>
        <p className="intro">Choose a depth-and-semantic sequence, an aligned opening image and a scene description. The model renders the recorded motion.</p>
        <label className="field">Official CWM examples<select aria-label="Official example" value={selected} onChange={(e) => chooseExample(e.target.value)} disabled={locked}>
          <option value="">Custom inputs / choose an example</option>
          {examples.map((x) => <option key={x.name} value={x.name}>{x.label}</option>)}
        </select></label>
        <p className="hint">40 examples from the official CWM release, supported by this model’s open-world entrypoint. Selecting one fills all inputs. Nothing starts automatically.</p>
        {catalogError && <p className="notice">{catalogError}</p>}
        <label className="upload"><span>01</span><div><strong>{filename(conditions, "Upload lossless conditions")}</strong><small>Depth + semantic IDs · ZIP · ≤512 MiB</small></div>
          <input aria-label="Conditions ZIP" type="file" accept=".zip,application/zip" disabled={locked} onChange={(e) => {
            if (e.target.files?.[0]) { setConditions(e.target.files[0]); setSelected(""); setPreview(null); } e.target.value = "";
          }} /></label>
        <label className="upload"><span>02</span><div><strong>{filename(image, "Upload reference image")}</strong><small>PNG / JPEG / WebP · aligned first frame · ≤16 MiB</small></div>
          <input aria-label="Reference image" type="file" accept="image/png,image/jpeg,image/webp" disabled={locked} onChange={(e) => {
            if (e.target.files?.[0]) { setImage(e.target.files[0]); setSelected(""); } e.target.value = "";
          }} /></label>
        <div className="appearance-preview"><h3>Appearance reference</h3>
          {imageUrl ? <img key={imageUrl} src={imageUrl} alt="Selected appearance reference" /> : <div className="empty">Choose an aligned opening image</div>}
        </div>
        <label className="field">03 / Appearance and scene<textarea aria-label="Rendering prompt" rows={7} maxLength={32768} disabled={locked} placeholder="Describe the starting scene and appearance…" value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
        <div className="numbers">
          <label className="field">Seed<input aria-label="Seed" type="number" min={0} max={2147483647} value={seed} disabled={locked} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          <label className="field">Chunk limit<input aria-label="Chunk limit" type="number" min={0} max={480} value={limit} disabled={locked} onChange={(e) => setLimit(Number(e.target.value))} /></label>
        </div>
        <p className="hint">17 frames per chunk. Choose 6–480 chunks, or 0 for all complete chunks. Prompt and reference stay fixed during each render. No live driving controls.</p>
        <button className="primary" onClick={active ? stop : render} disabled={Boolean(busy) || status !== "ready" || (!active && (!conditions || !image || !prompt.trim()))}>
          {active ? "Stop rendering" : world?.complete ? "Render again" : "Start rendering"}
        </button>
        <p className="progress" role="status">{progress}</p>
        {world?.complete && <p className="hint">{world.output_frames} frames published · {world.unused_tail_frames} source frames excluded by the limit or incomplete tail.</p>}
        {message && <div className="error" role="alert">{message}{message.includes("orphaned") && <p>The previous connection is closing. Wait about a minute before reconnecting.</p>}</div>}
        <details><summary className="hint">Custom condition format</summary><p className="hint">Consecutive pairs starting at 000000: .depth.f32 (336 × 192 little-endian float32 metres) and .semantic_id.png (336 × 192 grayscale IDs 0–11). At least 102 frames. A preview MP4 cannot replace these lossless inputs.</p></details>
      </aside>
      <section className="stage-column">
        <div className="stage-heading"><div><span className="eyebrow">02 / REFERENCE AND RENDER</span><h2>Structure in. A world out.</h2></div><span className="tag">24 FPS playback</span></div>
        <div className="view-labels"><h3>Reference video <span>Semantic conditions</span></h3><h3>Generated video <span>1344 × 768</span></h3></div>
        <div className="comparison">
          <div className="stage reference-stage">
            <ReactorView track="reference_video" muted videoObjectFit="contain" className="video" />
            {!world?.output_frames && <div className="placeholder"><span className="eyebrow">REFERENCE VIDEO</span><p>The matching semantic frames appear<br />when generated video is ready.</p></div>}
            {Boolean(world?.output_frames) && !referenceTrack && <div className="placeholder"><p>This Runtime provides generated video only.<br />Update Runtime to receive the reference track.</p></div>}
          </div>
          <div className="stage generated-stage">
            <ReactorView track="main_video" muted videoObjectFit="contain" className="video" />
            {!world?.output_frames && <div className="placeholder"><span className="eyebrow">{active ? "PREPARING FIRST FRAMES" : "READY WHEN YOU ARE"}</span><h2>{active ? "Conditioning your render" : "Choose an example. Press Start."}</h2><p>First-run text encoding and kernel compilation take time.<br />Native decoding needs one chunk of lookahead.</p></div>}
          </div>
        </div>
        <p className="stage-progress">Run {world?.world_id ?? "—"} · {world?.completed_chunks ?? 0} / {world?.total_chunks ?? "—"} decoded chunks · {world?.sampled_chunks ?? 0} sampled{world?.last_chunk_seconds != null && ` · ${world.last_chunk_seconds.toFixed(2)} s/step`}</p>
        <p className="hint">Matching reference and generated chunks stream together on separate video tracks. Playback may briefly differ while either track buffers. The left view visualizes semantic IDs; lossless depth and semantic pairs drive inference.</p>
        {preview && <details className="source-preview"><summary>Preview the full recorded input</summary><video key={preview} aria-label="Full semantic input preview" src={preview} controls muted preload="metadata" /><p className="hint">Independent source preview. Rendering starts at frame zero and stops at the selected chunk limit.</p></details>}
        <footer>Streaming Code World Model · TingtingLiao and contributors<span>Built on Reactor</span></footer>
      </section>
    </main>
  </div>;
}

function App() {
  const [endpoint, setEndpoint] = useState(() => new URLSearchParams(window.location.search).get("endpoint") ||
    import.meta.env.VITE_REACTOR_URL || "http://localhost:8080");
  return <ReactorProvider modelName="streaming-cwm" local apiUrl={endpoint.trim()}
    connectOptions={{ autoConnect: false }}>
    <Workspace endpoint={endpoint} onEndpointChange={setEndpoint} />
  </ReactorProvider>;
}

createRoot(document.querySelector("#root")).render(<App />);
