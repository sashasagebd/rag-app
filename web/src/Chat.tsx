// Questions and cited answers for one notebook. Each citation chip expands to the exact
// text the answer was based on. All text is rendered as plain text (never as HTML).
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, type Answer, type Citation } from "./api";

function Citations({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!citations.length) return null;
  const shown = citations.find((c) => c.n === open);
  return (
    <div className="citations">
      <div className="chips">
        {citations.map((c) => (
          <button key={c.n} className={`chip ${open === c.n ? "open" : ""}`} aria-expanded={open === c.n}
                  onClick={() => setOpen(open === c.n ? null : c.n)}>
            [{c.n}] {c.label}
          </button>
        ))}
      </div>
      {shown && <blockquote className="source">{shown.text}</blockquote>}
    </div>
  );
}

function AnswerView({ a }: { a: Answer }) {
  const how = a.route === "full" ? "read the whole notebook" : "searched the notebook";
  const cost = a.key_source === "cache" ? "cached, free" : `$${a.usd.toFixed(4)}${a.key_source === "server" ? " (shared key)" : ""}`;
  return (
    <div className="qa">
      <div className="bubble question">{a.question}</div>
      <div className={`bubble answer ${a.found ? "" : "not-found"}`}>
        <p>{a.answer}</p>
        <Citations citations={a.citations} />
        <div className="meta">{how} · {cost}</div>
      </div>
    </div>
  );
}

export default function Chat({ notebook, canAsk, onAsked }: {
  notebook: string;
  canAsk: boolean;
  onAsked: () => void;
}) {
  const [items, setItems] = useState<Answer[]>([]);
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setItems([]);
    setError("");
    api.questions(notebook).then(setItems);
  }, [notebook]);

  // Braces matter: an effect may only return a cleanup function, and in current browsers
  // scrollIntoView returns a Promise. Returning it crashed the page ("destroy is not a function").
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, pending]);

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    setPending(q);
    setQuestion("");
    setError("");
    try {
      const a = await api.ask(notebook, q);
      setItems((prev) => [...prev, a]);
      onAsked();
    } catch (err) {
      setError((err as Error).message);
      setQuestion(q); // give the question back so it can be retried
    } finally {
      setPending(null);
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <section className="chat">
      <div className="messages">
        {items.length === 0 && !pending && (
          <p className="muted empty">Ask anything about this notebook's documents. Answers cite their sources;
            if the documents don't say, you'll be told so.</p>
        )}
        {items.map((a) => <AnswerView key={a.id} a={a} />)}
        {pending && (
          <div className="qa">
            <div className="bubble question">{pending}</div>
            <div className="bubble answer thinking">Reading your documents…</div>
          </div>
        )}
        <div ref={bottom} />
      </div>
      {error && <p className="error">{error}</p>}
      <form className="ask" onSubmit={submit}>
        <textarea rows={2} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onKey}
                  placeholder={canAsk ? "Ask a question (Enter to send, Shift+Enter for a new line)"
                                      : "Add your API key in Settings to ask questions"}
                  disabled={!canAsk || pending !== null} maxLength={2000} />
        <button className="primary" disabled={!canAsk || pending !== null || !question.trim()}>Ask</button>
      </form>
    </section>
  );
}
