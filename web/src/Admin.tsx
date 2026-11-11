// Admin-only part of Settings: invite codes, and each user's server-key access and daily limit.
// The server checks admin rights on every call; hiding this panel is only for tidiness.
import { useEffect, useState, type FormEvent } from "react";
import { api, type AdminUser } from "./api";

export default function Admin() {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [note, setNote] = useState("");
  const [code, setCode] = useState("");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.adminUsers().then(setUsers).catch((e) => setError((e as Error).message));
  }, []);

  async function invite(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      setCode((await api.adminInvite(note.trim())).code);
      setCopied(false);
      setNote("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function copy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
  }

  async function update(username: string, patch: { use_server_key?: boolean; daily_question_limit?: number | null }) {
    setError("");
    try {
      setUsers(await api.adminUpdateUser(username, patch));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function editLimit(u: AdminUser) {
    const answer = prompt(`Questions per day for ${u.username} (empty = the default):`, u.daily_question_limit?.toString() ?? "");
    if (answer === null) return;
    const n = answer.trim() === "" ? null : Number(answer);
    if (n !== null && (!Number.isInteger(n) || n < 0)) return setError("the limit must be a whole number");
    update(u.username, { daily_question_limit: n });
  }

  return (
    <div className="card admin">
      <h3>Admin</h3>

      <form className="row" onSubmit={invite}>
        <input placeholder="Who it's for (optional)" value={note} onChange={(e) => setNote(e.target.value)} maxLength={100} />
        <button className="primary">Create invite</button>
      </form>
      {code && (
        <p className="invite-code">
          <code>{code}</code> <button className="link" onClick={copy}>{copied ? "copied" : "copy"}</button>
          <span className="hint"> One-time use. It won't be shown again.</span>
        </p>
      )}

      {users === null ? (
        <p className="muted">Loading users…</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>User</th><th>Joined</th><th>Own key</th><th>Server key</th><th>Limit/day</th><th>Today</th><th>Total $</th></tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.username}>
                  <td>{u.username}{u.is_admin && <span className="badge">admin</span>}</td>
                  <td>{u.created_at.slice(0, 10)}</td>
                  <td>{u.has_api_key ? "yes" : "no"}</td>
                  <td>
                    <button className="link" onClick={() => update(u.username, { use_server_key: !u.use_server_key })}>
                      {u.use_server_key ? "on" : "off"}
                    </button>
                  </td>
                  <td><button className="link" onClick={() => editLimit(u)}>{u.daily_question_limit ?? "default"}</button></td>
                  <td>{u.questions_today}</td>
                  <td>{u.usd_total.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
