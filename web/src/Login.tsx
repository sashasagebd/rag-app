// Log in, or sign up with a one-time invite code.
import { useState, type FormEvent } from "react";
import { api, type Me } from "./api";

export default function Login({ onDone }: { onDone: (me: Me) => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [invite, setInvite] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onDone(mode === "login" ? await api.login(username, password) : await api.signup(invite, username, password));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="card" onSubmit={submit}>
        <h1>Notebooks</h1>
        <p className="muted">Ask questions about your own documents. Every answer cites its source.</p>
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={mode === "login"} onClick={() => setMode("login")}>
            Log in
          </button>
          <button type="button" role="tab" aria-selected={mode === "signup"} onClick={() => setMode("signup")}>
            Sign up
          </button>
        </div>
        {mode === "signup" && (
          <label>
            Invite code
            <input value={invite} onChange={(e) => setInvite(e.target.value)} placeholder="xxxx-xxxx-xxxx" required />
          </label>
        )}
        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            minLength={mode === "signup" ? 8 : undefined}
            required
          />
        </label>
        {mode === "signup" && <p className="hint">At least 8 characters.</p>}
        {error && <p className="error">{error}</p>}
        <button className="primary" disabled={busy}>
          {busy ? "…" : mode === "login" ? "Log in" : "Create account"}
        </button>
      </form>
    </main>
  );
}
