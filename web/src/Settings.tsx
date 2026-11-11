// Your Anthropic API key, today's usage, and logging out.
import { useState, type FormEvent } from "react";
import Admin from "./Admin";
import { api, type Me } from "./api";

export default function Settings({ me, onChange, onLogout }: {
  me: Me;
  onChange: (me: Me) => void;
  onLogout: () => void;
}) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const u = me.usage;

  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onChange(await api.setApiKey(key.trim()));
      setKey("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm("Remove your API key? You won't be able to ask questions until you add one again.")) return;
    onChange(await api.deleteApiKey());
  }

  return (
    <section className="page">
      <h2>Settings</h2>

      <div className="card">
        <h3>Anthropic API key</h3>
        <p className="muted">
          Questions are answered with your own key and billed to your Anthropic account (about 0.05–0.2¢ per
          question). Create one at{" "}
          <a href="https://console.anthropic.com" target="_blank" rel="noreferrer">console.anthropic.com</a>.
          It is stored encrypted and never shown again.
        </p>
        {me.has_api_key ? (
          <p>
            Current key: <code>{me.api_key_hint}</code>{" "}
            <button className="link danger" onClick={remove}>remove</button>
          </p>
        ) : me.uses_server_key ? (
          <p>No key of your own: your questions use the shared key (it has a daily cap).</p>
        ) : (
          <p className="warn">No key yet: add one to ask questions.</p>
        )}
        <form className="row" onSubmit={save}>
          <input
            type="password"
            placeholder={me.has_api_key ? "Replace with a new key" : "sk-ant-…"}
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoComplete="off"
            required
          />
          <button className="primary" disabled={busy}>{busy ? "Checking…" : "Save"}</button>
        </form>
        {error && <p className="error">{error}</p>}
      </div>

      <div className="card">
        <h3>Usage</h3>
        <ul className="stats">
          <li>Questions today: <b>{u.questions_today}</b> of {u.questions_per_day} (resets at midnight UTC)</li>
          <li>Notebooks: <b>{u.notebooks}</b> of {u.max_notebooks}</li>
          <li>Storage: <b>{u.upload_mb} MB</b> of {u.max_upload_mb} MB</li>
        </ul>
      </div>

      {me.is_admin && <Admin />}

      <p>
        Logged in as <b>{me.username}</b>. <button className="link" onClick={onLogout}>Log out</button>
      </p>
    </section>
  );
}
