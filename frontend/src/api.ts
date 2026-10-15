export interface ApiError { error_code?: string; message: string; suggestion?: string | null; status?: string }
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try { res = await fetch(`/api${path}`, init); }
  catch { throw { message: "Cannot reach the backend. Is it running on port 8000?" } as ApiError; }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw { message: "Request failed", ...data } as ApiError;
  return data as T;
}
