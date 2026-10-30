/**
 * Hero photo: a signer in front of the translator. On large screens it fills the right of the hero
 * and fades into the page; on small screens it sits under the text as a rounded image.
 */
import heroImage from '../../assets/hero-signer.jpg';

export function HeroVisual() {
  return (
    <figure className="relative mx-4 sm:mx-6 mb-12 lg:m-0 lg:absolute lg:inset-y-0 lg:right-0 lg:w-[60%] lg:max-w-[1000px]">
      <img
        src={heroImage}
        alt="Illustration: a person signs “hello” to SignTalk AI, and the laptop shows the detected sign with its confidence"
        className="w-full aspect-[16/10] lg:aspect-auto lg:h-full object-cover object-right-top rounded-2xl lg:rounded-none"
      />
      {/* Fade the photo into the page, so it reads as part of the hero rather than a box */}
      <div className="hidden lg:block absolute inset-0 bg-linear-to-r from-bg via-bg/40 via-25% to-transparent to-55%" aria-hidden="true" />
      <div className="hidden lg:block absolute inset-x-0 bottom-0 h-1/4 bg-linear-to-t from-bg to-transparent" aria-hidden="true" />
      <div className="hidden lg:block absolute inset-x-0 top-0 h-16 bg-linear-to-b from-bg/70 to-transparent" aria-hidden="true" />
      <figcaption className="absolute top-3 right-4 lg:top-10 lg:right-10 -rotate-6 font-script text-right text-xl sm:text-2xl lg:text-[2rem] leading-tight text-white drop-shadow-[0_2px_6px_rgba(0,0,0,0.6)]">
        Different Hands
        <br />
        Same Language
        <br />
        A Brighter World
      </figcaption>
    </figure>
  );
}
