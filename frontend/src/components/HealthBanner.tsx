import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";

interface Status {
  ok: boolean;
  label: string;
  detail?: string;
}

export function HealthBanner() {
  const [backend, setBackend] = useState<Status>({ ok: false, label: "API: …" });
  const [model, setModel] = useState<Status>({ ok: false, label: "Model: …" });

  useEffect(() => {
    api
      .health()
      .then((h) =>
        setBackend({
          ok: h.status === "ok",
          label: `API: ${h.status}`,
          detail:
            h.bar_count != null ? `${h.bar_count.toLocaleString()} bars` : undefined,
        }),
      )
      .catch((e) =>
        setBackend({
          ok: false,
          label: "API: unreachable",
          detail: e instanceof ApiError ? e.detail : undefined,
        }),
      );

    api
      .modelHealth()
      .then((h) =>
        setModel({
          ok: h.model_loaded,
          label: h.model_loaded ? "Model: loaded" : "Model: not loaded",
          detail: h.error ?? undefined,
        }),
      )
      .catch((e) =>
        setModel({
          ok: false,
          label: "Model: offline",
          detail: e instanceof ApiError ? e.detail : undefined,
        }),
      );
  }, []);

  return (
    <div className="health">
      {[backend, model].map((s) => (
        <span
          key={s.label}
          className={`pill ${s.ok ? "pill-ok" : "pill-bad"}`}
          title={s.detail}
        >
          {s.label}
        </span>
      ))}
    </div>
  );
}
