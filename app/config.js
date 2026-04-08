/**
 * config.js — PathwayIQ frontend feature flags
 *
 * Loaded before app.js. Change values here to enable/disable features
 * without touching application code.
 */
const CONFIG = {
  // Persist saved courses to localStorage so they survive page refresh.
  // Set to false to keep saves in-memory only (e.g. pre-registration mode).
  SAVE_TO_LOCALSTORAGE: true,

  // Show a welcome-back advisory card on load when saved items exist in localStorage.
  // Only relevant when SAVE_TO_LOCALSTORAGE is true.
  SHOW_RETURN_REMINDER: true,
};
