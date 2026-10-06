import React from 'react';

export const CameraGuide: React.FC = () => {
  return (
    <div
      className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center p-6"
      aria-hidden="true"
    >
      {/* Subtle guide rectangle */}
      <div className="w-4/5 h-4/5 border-2 border-dashed border-teal-400/40 rounded-2xl relative flex flex-col items-center justify-between p-4">
        <span className="text-[11px] font-medium tracking-wide uppercase px-2 py-0.5 rounded bg-slate-900/60 text-teal-200">
          Position upper body & hands within frame
        </span>

        {/* Corner markers */}
        <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-teal-400 rounded-tl-lg" />
        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-teal-400 rounded-tr-lg" />
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-teal-400 rounded-bl-lg" />
        <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-teal-400 rounded-br-lg" />
      </div>
    </div>
  );
};
