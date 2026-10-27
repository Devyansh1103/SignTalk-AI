import { useState, type FormEvent } from 'react';
import { AlertCircle, CheckCircle2, GraduationCap, LoaderCircle, Send } from 'lucide-react';
import { CONTACT_ENDPOINT, PROJECT } from '../lib/project';
import { Button } from '../components/ui/Button';
import { GithubIcon } from '../components/ui/GithubIcon';

type Fields = { name: string; email: string; subject: string; message: string };
type Errors = Partial<Record<keyof Fields, string>>;

function validate(f: Fields): Errors {
  const e: Errors = {};
  if (f.name.trim().length < 2) e.name = 'Please enter your name.';
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(f.email.trim())) e.email = 'Please enter a valid email address.';
  if (f.subject.trim().length < 3) e.subject = 'Please add a short subject.';
  if (f.message.trim().length < 10) e.message = 'Your message should be at least 10 characters.';
  return e;
}

/** Without a configured endpoint, the form hands the message to a prefilled GitHub issue instead. */
function githubIssueUrl(f: Fields) {
  const body = `${f.message.trim()}\n\n— ${f.name.trim()}`;
  return `${PROJECT.repoUrl}/issues/new?title=${encodeURIComponent(f.subject.trim())}&body=${encodeURIComponent(body)}`;
}

export function ContactPage() {
  const [fields, setFields] = useState<Fields>({ name: '', email: '', subject: '', message: '' });
  const [errors, setErrors] = useState<Errors>({});
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'github' | 'failed'>('idle');

  const set = (k: keyof Fields) => (e: { target: { value: string } }) => {
    setFields((f) => ({ ...f, [k]: e.target.value }));
    if (errors[k]) setErrors((x) => ({ ...x, [k]: undefined }));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const found = validate(fields);
    setErrors(found);
    if (Object.keys(found).length) {
      document.getElementById(`contact-${Object.keys(found)[0]}`)?.focus();
      return;
    }
    if (!CONTACT_ENDPOINT) {
      window.open(githubIssueUrl(fields), '_blank', 'noopener');
      setState('github');
      return;
    }
    setState('sending');
    try {
      const res = await fetch(CONTACT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(fields),
      });
      if (!res.ok) throw new Error(String(res.status));
      setState('sent');
      setFields({ name: '', email: '', subject: '', message: '' });
    } catch {
      setState('failed');
    }
  };

  const input = 'w-full px-3.5 py-2.5 rounded-xl bg-surface-2 border text-sm text-ink placeholder:text-ink-3';
  const field = (k: keyof Fields, label: string, props: Record<string, unknown> = {}, area = false) => {
    const id = `contact-${k}`;
    const common = {
      id,
      value: fields[k],
      onChange: set(k),
      'aria-invalid': !!errors[k],
      'aria-describedby': errors[k] ? `${id}-error` : undefined,
      className: `${input} ${errors[k] ? 'border-bad' : 'border-line'}`,
      ...props,
    };
    return (
      <div>
        <label htmlFor={id} className="block text-xs font-medium text-ink-2 mb-1.5">
          {label}
        </label>
        {area ? <textarea rows={5} {...common} /> : <input {...common} />}
        {errors[k] && (
          <p id={`${id}-error`} className="text-xs text-bad mt-1">
            {errors[k]}
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 grid lg:grid-cols-2 gap-10 items-start">
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-accent-text mb-2">Contact</p>
        <h1 className="text-4xl font-bold tracking-tight text-ink">Get in Touch</h1>
        <p className="mt-4 text-ink-2 leading-relaxed max-w-lg">
          Questions, suggestions, or a sign that is not recognised well? Send a message; feedback helps improve the model.
        </p>

        <ul className="grid sm:grid-cols-2 gap-4 mt-8">
          <li>
            <a href={PROJECT.repoUrl} target="_blank" rel="noreferrer" className="card p-5 flex items-start gap-4 hover:border-line-strong h-full">
              <span className="w-11 h-11 rounded-xl bg-accent-soft text-accent-text flex items-center justify-center shrink-0">
                <GithubIcon className="w-5 h-5" />
              </span>
              <span>
                <span className="block text-sm font-semibold text-ink">GitHub</span>
                <span className="block text-xs text-ink-2 mt-0.5">View the project repository</span>
              </span>
            </a>
          </li>
          <li className="card p-5 flex items-start gap-4">
            <span className="w-11 h-11 rounded-xl bg-accent-soft text-accent-text flex items-center justify-center shrink-0">
              <GraduationCap className="w-5 h-5" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-sm font-semibold text-ink">Project</span>
              <span className="block text-xs text-ink-2 mt-0.5">
                {PROJECT.programme}
                <br />
                {PROJECT.institution}
              </span>
            </span>
          </li>
        </ul>
      </div>

      <div className="relative">
      <div className="absolute -inset-6 rounded-[2rem] bg-gradient-accent opacity-50 blur-3xl" aria-hidden="true" />
      <section aria-labelledby="form-title" className="relative card p-6">
        <h2 id="form-title" className="text-lg font-semibold text-ink mb-5">
          Send a Message
        </h2>
        <form noValidate onSubmit={submit} className="flex flex-col gap-4">
          <div className="grid sm:grid-cols-2 gap-4">
            {field('name', 'Your name', { placeholder: 'Enter your name', autoComplete: 'name' })}
            {field('email', 'Your email', { type: 'email', placeholder: 'Enter your email', autoComplete: 'email' })}
          </div>
          {field('subject', 'Subject', { placeholder: 'Enter subject' })}
          {field('message', 'Message', { placeholder: 'Type your message here…' }, true)}

          <Button type="submit" variant="primary" size="lg" disabled={state === 'sending'} className="w-full">
            {state === 'sending' ? <LoaderCircle className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            {state === 'sending' ? 'Sending…' : CONTACT_ENDPOINT ? 'Send Message' : 'Continue on GitHub'}
          </Button>

          <div aria-live="polite">
            {state === 'sent' && (
              <p className="flex items-start gap-2 text-sm text-ok">
                <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
                Thank you! Your message has been sent.
              </p>
            )}
            {state === 'failed' && (
              <p className="flex items-start gap-2 text-sm text-bad">
                <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
                The message could not be sent. Please try again, or open an issue on GitHub.
              </p>
            )}
            {state === 'github' && (
              <p className="flex items-start gap-2 text-sm text-ink-2">
                <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0 text-ok" aria-hidden="true" />
                A prefilled GitHub issue opened in a new tab. Press “Submit new issue” there to send it.
              </p>
            )}
          </div>
          {!CONTACT_ENDPOINT && (
            <p className="text-xs text-ink-3">
              Messages are posted as a public GitHub issue (a GitHub account is needed). Your email address is not included.
            </p>
          )}
        </form>
      </section>
      </div>
    </div>
  );
}
