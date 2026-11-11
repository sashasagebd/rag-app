// Every call to the backend, with the shapes it returns (mirrors src/api/app.py).
// Errors come back as ApiError with the server's readable message in `.message`.

export type Usage = {
  questions_today: number;
  questions_per_day: number;
  notebooks: number;
  max_notebooks: number;
  upload_mb: number;
  max_upload_mb: number;
};

export type Me = {
  username: string;
  has_api_key: boolean;
  api_key_hint: string | null;
  uses_server_key: boolean;
  is_admin: boolean;
  usage: Usage;
};

export type AdminUser = {
  username: string;
  created_at: string;
  has_api_key: boolean;
  use_server_key: boolean;
  daily_question_limit: number | null; // null = the default
  questions_today: number;
  usd_total: number;
  is_admin: boolean;
};

export type Notebook = { name: string; documents: number; tokens: number; created_at: string };

export type Doc = {
  name: string;
  source: string;
  doc_type: "pdf" | "html";
  n_bytes: number;
  n_chunks: number;
  n_tokens: number;
  added_at: string;
  status: "processing" | "ready" | "failed";
  error: string | null;
};

export type Citation = {
  n: number;
  chunk_key: string;
  doc: string;
  label: string; // "paper.pdf, p. 4" or "slides.html, Intro > Grading"
  sections: string[];
  pages: number[];
  text: string;
};

export type Answer = {
  id: number;
  question: string;
  answer: string;
  found: boolean;
  route: "full" | "retrieval";
  citations: Citation[];
  key_source: "user" | "server" | "cache";
  usd: number;
  created_at: string;
};

export type AddResult = { name: string; status: "processing" | "skipped"; reason?: string };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const isForm = body instanceof FormData;
  const res = await fetch(`/api${path}`, {
    method,
    credentials: "same-origin",
    headers: body === undefined || isForm ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const nb = (name: string) => `/notebooks/${encodeURIComponent(name)}`;

export const api = {
  me: () => request<Me>("GET", "/me"),
  login: (username: string, password: string) => request<Me>("POST", "/auth/login", { username, password }),
  signup: (invite: string, username: string, password: string) =>
    request<Me>("POST", "/auth/signup", { invite, username, password }),
  logout: () => request<void>("POST", "/auth/logout"),
  setApiKey: (api_key: string) => request<Me>("PUT", "/me/api-key", { api_key }),
  deleteApiKey: () => request<Me>("DELETE", "/me/api-key"),

  notebooks: () => request<Notebook[]>("GET", "/notebooks"),
  createNotebook: (name: string) => request<{ name: string }>("POST", "/notebooks", { name }),
  deleteNotebook: (name: string) => request<void>("DELETE", nb(name)),

  documents: (notebook: string) => request<Doc[]>("GET", `${nb(notebook)}/documents`),
  upload: (notebook: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<AddResult>("POST", `${nb(notebook)}/documents`, form);
  },
  addUrl: (notebook: string, url: string) => request<AddResult>("POST", `${nb(notebook)}/documents/url`, { url }),
  deleteDocument: (notebook: string, doc: string) =>
    request<void>("DELETE", `${nb(notebook)}/documents/${encodeURIComponent(doc)}`),

  questions: (notebook: string) => request<Answer[]>("GET", `${nb(notebook)}/questions`),
  ask: (notebook: string, question: string) => request<Answer>("POST", `${nb(notebook)}/ask`, { question }),

  // admins only
  adminUsers: () => request<AdminUser[]>("GET", "/admin/users"),
  adminInvite: (note: string) => request<{ code: string }>("POST", "/admin/invites", { note }),
  adminUpdateUser: (username: string, patch: { use_server_key?: boolean; daily_question_limit?: number | null }) =>
    request<AdminUser[]>("PATCH", `/admin/users/${encodeURIComponent(username)}`, patch),
};
