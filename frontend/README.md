# SignTalk AI — frontend v2

Redesigned web app (Home, Live Translator, About, Contact; dark and light themes). It sits next to
`../frontend`, which is unchanged. The recognition code (`src/engine`, `src/hooks`, `src/services`,
`src/types`) is copied from `../frontend`. The AI models are served from `../frontend/public`, so
both apps always use the same model.

```
cd frontend-v2
npm install
npm run dev        # http://localhost:5174
npm run build      # production build in dist/
```

- `VITE_INFERENCE=server` uses the FastAPI backend instead of in-browser inference, same as `../frontend`.
- `VITE_CONTACT_ENDPOINT` is optional. It is a form service URL (for example Formspree) that accepts a
  JSON POST of `{name, email, subject, message}`. Without it, the Contact form opens a prefilled GitHub
  issue instead.
- Pages use hash routes (`#/translator`), so any static host works without rewrite rules. On Vercel,
  set the Root Directory to `frontend-v2`.
