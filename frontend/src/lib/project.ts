/** Facts shown on the site. Keep these to what the repository can back up. */
export const PROJECT = {
  name: 'SignTalk AI',
  tagline: 'Real-time Sign Language Translation',
  institution: 'GLA University',
  programme: 'B.Tech CSE (AI/ML) Academic Project',
  repoUrl: 'https://github.com/Devyansh1103/SignTalk-AI',
  dataset: 'INCLUDE (Indian Lexicon Sign Language Dataset)',
  // Measured with scripts/evaluate_robustness.py on the INCLUDE test split (results/stgcn_colab_v4/robustness.txt)
  testAccuracy: '87.6%',
  testClips: 645,
};

/**
 * Optional form endpoint (e.g. Formspree) that accepts a JSON POST of {name, email, subject, message}.
 * Without it the contact form opens a prefilled GitHub issue instead of pretending to send mail.
 */
export const CONTACT_ENDPOINT: string | undefined = import.meta.env.VITE_CONTACT_ENDPOINT || undefined;
