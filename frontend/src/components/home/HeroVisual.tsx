/**
 * Illustration of the translator: a hand drawn with MediaPipe's 21-point hand topology over a
 * camera frame, next to a result card. Clearly marked as an example, not live output.
 */

const HAND: [number, number][] = [
  [100, 190], [72, 172], [54, 150], [42, 128], [32, 108],
  [80, 122], [74, 90], [70, 68], [67, 46],
  [100, 116], [100, 80], [100, 56], [100, 32],
  [120, 120], [124, 88], [127, 66], [129, 46],
  [138, 132], [147, 108], [152, 92], [156, 76],
];
const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export function HeroVisual() {
  return (
    <figure className="relative" aria-label="Illustration of the Live Translator recognising a sign">
      <div className="absolute -inset-6 rounded-[2rem] bg-gradient-accent opacity-50 blur-3xl" aria-hidden="true" />
      <div className="relative card p-3 sm:p-4">
        <div className="flex items-center justify-between px-1 pb-3">
          <div className="flex gap-1.5" aria-hidden="true">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-400/70" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400/70" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400/70" />
          </div>
          <span className="text-[11px] text-ink-3">Example</span>
        </div>

        <div className="grid grid-cols-5 gap-3">
          {/* Camera frame with landmarks */}
          <div className="col-span-3 relative aspect-[4/5] rounded-xl overflow-hidden bg-[radial-gradient(circle_at_50%_35%,#1d2a5c,#070c1f_70%)]">
            <svg viewBox="0 0 200 250" className="absolute inset-0 w-full h-full" aria-hidden="true">
              <defs>
                <linearGradient id="bone" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#7c8cff" />
                  <stop offset="100%" stopColor="#a78bfa" />
                </linearGradient>
              </defs>
              {/* Shoulders and head, as the pose landmarks see them */}
              <path d="M20 250 C 30 205, 60 195, 100 195 C 140 195, 170 205, 180 250" fill="#ffffff0d" stroke="#ffffff1f" />
              <circle cx="100" cy="150" r="30" fill="#ffffff0a" stroke="#ffffff1f" />
              <g transform="translate(20 0) scale(0.9)">
                {BONES.map(([a, b]) => (
                  <line key={`${a}-${b}`} x1={HAND[a][0]} y1={HAND[a][1]} x2={HAND[b][0]} y2={HAND[b][1]} stroke="url(#bone)" strokeWidth="2.2" strokeLinecap="round" />
                ))}
                {HAND.map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r={i === 0 ? 4 : 3} fill="#eef2ff" stroke="#6d7dff" strokeWidth="1.5" />
                ))}
              </g>
            </svg>
            <span className="absolute top-2 left-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/90 text-white text-[10px] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-white" />
              Live
            </span>
          </div>

          {/* Result card */}
          <div className="col-span-2 flex flex-col gap-3 min-w-0">
            <div className="rounded-xl bg-surface-2 border border-line p-3">
              <p className="text-[10px] text-ink-3 mb-1">Detected sign</p>
              <p className="text-xl font-bold text-ink leading-tight">Hello</p>
              <p className="text-sm text-accent-text">नमस्ते</p>
            </div>
            <div className="rounded-xl bg-surface-2 border border-line p-3">
              <p className="text-[10px] text-ink-3 mb-2">Gesture sequence</p>
              <div className="flex flex-wrap gap-1">
                {['Hello', 'Teacher'].map((w) => (
                  <span key={w} className="px-1.5 py-0.5 rounded-md bg-accent-soft text-accent-text text-[10px] font-medium">
                    {w}
                  </span>
                ))}
              </div>
            </div>
            <div className="rounded-xl bg-surface-2 border border-line p-3 flex-1">
              <p className="text-[10px] text-ink-3 mb-1">Sentence</p>
              <p className="text-xs font-medium text-ink">Hello teacher!</p>
              <p className="text-xs text-accent-text">नमस्ते शिक्षक!</p>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="sr-only">Example only. Real results appear on the Live Translator page.</figcaption>
    </figure>
  );
}
