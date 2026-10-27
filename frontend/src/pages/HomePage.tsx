import { Accessibility, ArrowDown, ArrowRight, Camera, Cpu, Hand, Languages, Network, Play, ScanLine, Type, Zap } from 'lucide-react';
import { hrefFor } from '../lib/useRoute';
import { buttonClass } from '../components/ui/buttonClass';
import { HeroVisual } from '../components/home/HeroVisual';

const STEPS = [
  { icon: Camera, title: 'Capture', text: 'Your webcam video is read frame by frame in the browser.' },
  { icon: ScanLine, title: 'Landmarks', text: 'MediaPipe finds the hand and upper-body landmarks in each frame.' },
  { icon: Network, title: 'ST-GCN', text: 'When you lower your hands, the sign’s landmark sequence goes to the ST-GCN model.' },
  { icon: Languages, title: 'Output', text: 'The recognised word, its confidence and the sentence appear in English and Hindi.' },
];

export function HomePage({ classes }: { classes: number }) {
  const words = classes ? `${classes} ISL words` : 'ISL words';
  const features = [
    { icon: Hand, title: 'Live Gesture Recognition', text: 'MediaPipe hand and pose landmarks, tracked live from your webcam.' },
    { icon: Type, title: 'English & Hindi Text', text: 'Each sign and sentence is shown as text in both languages.' },
    { icon: Cpu, title: 'ST-GCN Model', text: `A graph neural network trained on ${words} from the INCLUDE dataset.` },
    { icon: Accessibility, title: 'Accessible Communication', text: 'Large text, high contrast and keyboard shortcuts built in.' },
  ];

  return (
    <>
      {/* Hero */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 lg:pt-16 pb-12 grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <div className="relative w-fit mb-6">
            <div className="absolute -inset-2 rounded-full bg-gradient-accent opacity-50 blur-xl" aria-hidden="true" />
            <p className="relative inline-flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-1.5 rounded-full border border-line bg-surface text-xs text-ink-2">
              <Zap className="w-3.5 h-3.5 text-accent-text" aria-hidden="true" />
              Real-time <span aria-hidden="true">·</span> AI Powered <span aria-hidden="true">·</span> Indian Sign Language (ISL)
            </p>
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.05] text-ink">
            Breaking Barriers <span className="block w-fit text-gradient">Through AI</span>
          </h1>
          <p className="mt-6 text-base sm:text-lg text-ink-2 max-w-xl leading-relaxed">
            SignTalk AI translates Indian Sign Language gestures into English and Hindi text in real time, making communication more
            inclusive and accessible.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href={hrefFor('translator')} className={buttonClass('primary', 'lg')}>
              <Play className="w-4 h-4 fill-current" aria-hidden="true" />
              Try Live Translator
              <ArrowRight className="w-4 h-4" aria-hidden="true" />
            </a>
            <button
              type="button"
              className={buttonClass('secondary', 'lg')}
              onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ block: 'start' })}
            >
              Learn More
              <ArrowDown className="w-4 h-4" aria-hidden="true" />
            </button>
          </div>
          <ul className="mt-10 grid grid-cols-3 gap-4 max-w-lg">
            {[
              { icon: Languages, text: 'Real-time translation' },
              { icon: Hand, text: 'Indian Sign Language' },
              { icon: Accessibility, text: 'Runs in your browser' },
            ].map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-center gap-2.5 text-xs text-ink-2">
                <span className="w-9 h-9 rounded-full border border-line bg-surface flex items-center justify-center text-accent-text shrink-0">
                  <Icon className="w-4 h-4" aria-hidden="true" />
                </span>
                {text}
              </li>
            ))}
          </ul>
        </div>
        <HeroVisual />
      </section>

      {/* Features */}
      <section aria-label="Features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ul className="card grid sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 lg:divide-x divide-line">
          {features.map(({ icon: Icon, title, text }) => (
            <li key={title} className="flex items-start gap-4 p-5">
              <span className="w-11 h-11 rounded-xl bg-accent-soft text-accent-text flex items-center justify-center shrink-0">
                <Icon className="w-5 h-5" aria-hidden="true" />
              </span>
              <div>
                <h2 className="text-sm font-semibold text-ink">{title}</h2>
                <p className="text-xs text-ink-2 mt-1 leading-relaxed">{text}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      {/* How it works */}
      <section id="how-it-works" aria-labelledby="how-title" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 scroll-mt-20">
        <div className="text-center mb-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-accent-text mb-2">How it works</p>
          <h2 id="how-title" className="text-3xl font-bold tracking-tight text-ink">
            From <span className="text-gradient">Gesture</span> to <span className="text-gradient">Meaning</span>
          </h2>
          <p className="text-sm text-ink-2 mt-3">Four steps, all running on your own device.</p>
        </div>
        <ol className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {STEPS.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="card p-5 relative">
              <div className="flex items-center gap-3 mb-3">
                <span className="w-11 h-11 rounded-full bg-gradient-accent text-white flex items-center justify-center shrink-0">
                  <Icon className="w-5 h-5" aria-hidden="true" />
                </span>
                <span className="text-xs font-semibold text-ink-3">Step {i + 1}</span>
              </div>
              <h3 className="text-sm font-semibold text-ink">{title}</h3>
              <p className="text-xs text-ink-2 mt-1 leading-relaxed">{text}</p>
              {i < STEPS.length - 1 && (
                <ArrowRight className="hidden lg:block absolute -right-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-ink-3 bg-bg rounded-full" aria-hidden="true" />
              )}
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}
