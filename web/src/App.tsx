// The app shell: log-in gate, sidebar with notebooks, and the open page.
import { useEffect, useState, type FormEvent } from "react";
import { api, ApiError, type Me, type Notebook } from "./api";
import Chat from "./Chat";
import Documents from "./Documents";
import Login from "./Login";
import Settings from "./Settings";

type View = { kind: "notebook"; name: string } | { kind: "settings" } | { kind: "home" };

export default function App() {
  const [me, setMe] = useState<Me | null | undefined>(undefined); // undefined = still checking
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [view, setView] = useState<View>({ kind: "home" });
  const [newName, setNewName] = useState("");
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [offline, setOffline] = useState("");

  const refreshMe = () =>
    api.me()
      .then((m) => { setMe(m); setOffline(""); })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setMe(null);
        else setOffline(`Can't reach the server (${(e as Error).message}). Try reloading in a moment.`);
      });
  const refreshNotebooks = () => api.notebooks().then(setNotebooks);

  useEffect(() => { refreshMe(); }, []);
  useEffect(() => {
    if (!me) return;
    refreshNotebooks();
    if (!me.has_api_key && !me.uses_server_key) setView({ kind: "settings" }); // first thing a new user needs
  }, [me?.username]);

  if (me === undefined) return <p className={offline ? "error center" : "muted center"}>{offline || "Loading…"}</p>;
  if (me === null) return <Login onDone={setMe} />;

  const open = (v: View) => { setView(v); setMenuOpen(false); };

  async function create(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const { name } = await api.createNotebook(newName.trim().toLowerCase());
      setNewName("");
      await refreshNotebooks();
      refreshMe();
      open({ kind: "notebook", name });
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function remove(name: string) {
    if (!confirm(`Delete the notebook "${name}" and all its documents and questions? This can't be undone.`)) return;
    await api.deleteNotebook(name);
    await refreshNotebooks();
    refreshMe();
    open({ kind: "home" });
  }

  async function logout() {
    await api.logout();
    setMe(null);
    setNotebooks([]);
    setView({ kind: "home" });
  }

  const canAsk = me.has_api_key || me.uses_server_key;

  return (
    <div className="app">
      <header className="topbar">
        <button className="link menu" onClick={() => setMenuOpen(!menuOpen)} aria-label="Notebooks">☰</button>
        <span className="brand">Notebooks</span>
        <span className="spacer" />
        <span className="muted small">{me.usage.questions_today}/{me.usage.questions_per_day} questions today</span>
        <button className="link" onClick={() => open({ kind: "settings" })}>{me.username}</button>
      </header>

      <nav className={`sidebar ${menuOpen ? "open" : ""}`}>
        <ul>
          {notebooks.map((n) => (
            <li key={n.name}>
              <button className={view.kind === "notebook" && view.name === n.name ? "active" : ""}
                      onClick={() => open({ kind: "notebook", name: n.name })}>
                {n.name} <span className="muted small">{n.documents}</span>
              </button>
            </li>
          ))}
        </ul>
        <form onSubmit={create}>
          <input placeholder="new-notebook" value={newName} onChange={(e) => setNewName(e.target.value)}
                 pattern="[a-z0-9][a-z0-9\-]*" title="lowercase letters, digits and dashes" maxLength={50} required />
          <button>Create</button>
        </form>
        {error && <p className="error small">{error}</p>}
        <button className="link settings-link" onClick={() => open({ kind: "settings" })}>Settings</button>
      </nav>

      <main className="content">
        {view.kind === "settings" && <Settings me={me} onChange={setMe} onLogout={logout} />}
        {view.kind === "home" && (
          <section className="page">
            <h2>Welcome, {me.username}</h2>
            <p className="muted">
              {notebooks.length ? "Pick a notebook on the left, or create a new one."
                                : "Create a notebook on the left, then add PDFs or web pages to it."}
            </p>
            {!canAsk && <p className="warn">Add your Anthropic API key in <button className="link" onClick={() => open({ kind: "settings" })}>Settings</button> to ask questions.</p>}
          </section>
        )}
        {view.kind === "notebook" && (
          <div className="notebook">
            <div className="notebook-head">
              <h2>{view.name}</h2>
              <button className="link danger" onClick={() => remove(view.name)}>delete notebook</button>
            </div>
            <div className="notebook-body">
              <Documents notebook={view.name} onChange={refreshNotebooks} />
              <Chat notebook={view.name} canAsk={canAsk} onAsked={refreshMe} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
