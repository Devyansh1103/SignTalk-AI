import { ArrowRight, CheckCircle2, Cpu, Info, Layers } from 'lucide-react';
import { PROJECT } from '../lib/project';

const STACK = [
  { group: 'Model & training', items: ['Python', 'PyTorch', 'ST-GCN', 'MediaPipe', 'OpenCV'] },
  { group: 'Web app', items: ['React', 'TypeScript', 'Vite', 'Tailwind CSS', 'ONNX Runtime Web'] },
  { group: 'Optional server', items: ['FastAPI', 'WebSocket'] },
];

const PIPELINE = ['Video frame', 'MediaPipe landmarks', 'Sign segment', 'ST-GCN model', 'Sign + sentence'];

export function AboutPage({ classes }: { classes: number }) {
  const words = classes || 260;
  const features = [
    `Recognises ${words} Indian Sign Language words`,
    'English and Hindi output for every sign',
    'Joins consecutive signs into simple sentences',
    'Runs entirely in the browser; video never leaves the device',
    'Camera selection, mirrored preview and keyboard shortcuts',
    'Large Text and High Contrast modes, dark and light themes',
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14">
      <div className="grid lg:grid-cols-2 gap-10 items-start">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-accent-text mb-2">About</p>
          <h1 className="text-4xl font-bold tracking-tight text-ink">Our Project</h1>
          <div className="mt-5 flex flex-col gap-4 text-ink-2 leading-relaxed">
            <p>
              SignTalk AI is a {PROJECT.programme.replace(' Academic Project', '')} academic project at {PROJECT.institution}. It
              translates Indian Sign Language (ISL) signs, captured with an ordinary webcam, into English and Hindi text in real
              time.
            </p>
            <p>
              Deaf and hard-of-hearing people who sign in ISL often depend on an interpreter to talk with people who do not know
              the language, and interpreters are not always available. SignTalk AI is a step towards everyday tools that let both
              sides understand each other directly.
            </p>
          </div>
        </div>

        <div className="relative">
        <div className="absolute -inset-6 rounded-[2rem] bg-gradient-accent opacity-50 blur-3xl" aria-hidden="true" />
        <div className="relative card p-6">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-ink mb-4">
            <Info className="w-4 h-4 text-accent-text" aria-hidden="true" />
            At a glance
          </h2>
          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div className="rounded-xl bg-surface-2 border border-line p-4">
              <dt className="text-xs text-ink-3">Vocabulary</dt>
              <dd className="text-2xl font-bold text-ink mt-1">{words}</dd>
              <dd className="text-xs text-ink-2">ISL words</dd>
            </div>
            <div className="rounded-xl bg-surface-2 border border-line p-4">
              <dt className="text-xs text-ink-3">Test accuracy</dt>
              <dd className="text-2xl font-bold text-ink mt-1">{PROJECT.testAccuracy}</dd>
              <dd className="text-xs text-ink-2">top-1, {PROJECT.testClips} INCLUDE test clips</dd>
            </div>
            <div className="col-span-2 rounded-xl bg-surface-2 border border-line p-4">
              <dt className="text-xs text-ink-3">Training data</dt>
              <dd className="text-sm text-ink mt-1">{PROJECT.dataset}, plus webcam recordings made for this project</dd>
            </div>
          </dl>
        </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-5 mt-10">
        <section aria-labelledby="arch-title" className="card p-6 lg:col-span-2">
          <h2 id="arch-title" className="flex items-center gap-2 text-sm font-semibold text-ink mb-4">
            <Cpu className="w-4 h-4 text-accent-text" aria-hidden="true" />
            Model architecture
          </h2>
          <ol className="flex flex-wrap items-center gap-2 mb-5">
            {PIPELINE.map((step, i) => (
              <li key={step} className="flex items-center gap-2">
                <span className="px-3 py-2 rounded-xl bg-surface-2 border border-line text-xs font-medium text-ink">{step}</span>
                {i < PIPELINE.length - 1 && <ArrowRight className="w-4 h-4 text-ink-3" aria-hidden="true" />}
              </li>
            ))}
          </ol>
          <div className="flex flex-col gap-3 text-sm text-ink-2 leading-relaxed">
            <p>
              <strong className="text-ink">MediaPipe</strong> finds 21 landmarks on each hand and key upper-body points in every
              frame. Positions are normalised to the shoulders, so it does not matter where you stand in the frame.
            </p>
            <p>
              A sign starts when your hands come up and ends when they go back down. That stretch of frames is resampled to a fixed
              length and passed to a <strong className="text-ink">Spatial-Temporal Graph Convolutional Network (ST-GCN)</strong>,
              which treats the landmarks as a graph of joints and learns how they move together over time.
            </p>
            <p>
              A sign is accepted only when the model is confident enough; recognised signs are then joined into a sentence after a
              short pause.
            </p>
          </div>
        </section>

        <section aria-labelledby="stack-title" className="card p-6">
          <h2 id="stack-title" className="flex items-center gap-2 text-sm font-semibold text-ink mb-4">
            <Layers className="w-4 h-4 text-accent-text" aria-hidden="true" />
            Technology stack
          </h2>
          <div className="flex flex-col gap-4">
            {STACK.map(({ group, items }) => (
              <div key={group}>
                <h3 className="text-xs text-ink-3 mb-2">{group}</h3>
                <ul className="flex flex-wrap gap-1.5">
                  {items.map((t) => (
                    <li key={t} className="px-2.5 py-1 rounded-lg bg-surface-2 border border-line text-xs text-ink">
                      {t}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section aria-labelledby="features-title" className="card p-6 mt-5">
        <h2 id="features-title" className="text-sm font-semibold text-ink mb-4">
          Key features
        </h2>
        <ul className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {features.map((f) => (
            <li key={f} className="flex items-start gap-2.5 text-sm text-ink-2">
              <CheckCircle2 className="w-4 h-4 text-ok shrink-0 mt-0.5" aria-hidden="true" />
              {f}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
