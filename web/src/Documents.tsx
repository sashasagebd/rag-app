// A notebook's documents: upload files or add links, watch processing, remove.
import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type Doc } from "./api";

const fmtMB = (bytes: number) => `${(bytes / 1e6).toFixed(bytes < 1e5 ? 2 : 1)} MB`;

export default function Documents({ notebook, onChange }: { notebook: string; onChange: () => void }) {
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<{ text: string; error: boolean }[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = () => api.documents(notebook).then(setDocs);

  useEffect(() => {
    setDocs(null);
    setMessages([]);
    load();
  }, [notebook]);

  // While anything is processing, check again every 2 seconds.
  const processing = docs?.some((d) => d.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => load().then(onChange), 2000);
    return () => clearInterval(timer);
  }, [processing, notebook]);

  async function add(task: () => Promise<{ name: string; status: string; reason?: string }>, label: string) {
    try {
      const r = await task();
      if (r.status === "skipped") setMessages((m) => [...m, { text: `${label}: skipped, ${r.reason}`, error: false }]);
    } catch (err) {
      setMessages((m) => [...m, { text: `${label}: ${(err as Error).message}`, error: true }]);
    }
  }

  async function uploadFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setMessages([]);
    for (const file of Array.from(files)) await add(() => api.upload(notebook, file), file.name);
    if (fileInput.current) fileInput.current.value = "";
    setBusy(false);
    await load();
    onChange();
  }

  async function addLink(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMessages([]);
    await add(() => api.addUrl(notebook, url.trim()), url.trim());
    setUrl("");
    setBusy(false);
    await load();
    onChange();
  }

  async function remove(doc: Doc) {
    if (!confirm(`Remove ${doc.name} from this notebook?`)) return;
    await api.deleteDocument(notebook, doc.name);
    await load();
    onChange();
  }

  return (
    <aside className="documents">
      <h3>Documents</h3>
      <label className="upload">
        <input ref={fileInput} type="file" accept=".pdf,.html,.htm" multiple disabled={busy}
               onChange={(e) => uploadFiles(e.target.files)} />
        <span>{busy ? "Adding…" : "Upload PDF or HTML"}</span>
      </label>
      <form className="row" onSubmit={addLink}>
        <input type="url" placeholder="or paste a link" value={url} onChange={(e) => setUrl(e.target.value)}
               disabled={busy} required />
        <button disabled={busy}>Add</button>
      </form>
      {messages.map((m, i) => <p key={i} className={m.error ? "error" : "hint"}>{m.text}</p>)}

      {docs === null ? (
        <p className="muted">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="muted">No documents yet. Add a PDF or a web page to start asking questions.</p>
      ) : (
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.name} className={d.status}>
              <div className="doc-name" title={d.source}>{d.name}</div>
              <div className="doc-meta">
                {d.status === "processing" && <span className="badge">processing…</span>}
                {d.status === "failed" && <span className="badge failed">failed</span>}
                {d.status === "ready" && <>{d.doc_type.toUpperCase()} · {fmtMB(d.n_bytes)} · {d.n_chunks} chunks</>}
                <button className="link danger" onClick={() => remove(d)}>remove</button>
              </div>
              {d.error && <div className="error small">{d.error}</div>}
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
