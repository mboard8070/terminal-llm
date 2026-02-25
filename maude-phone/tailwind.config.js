/** @type {import('tailwindcss').Config} */

export default {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--maude-font)", "Inter", "sans-serif"],
        mono: ["VT323", "Share Tech Mono", "monospace"],
      },
      colors: {
        maude: {
          bg: "var(--maude-bg)",
          surface: "var(--maude-surface)",
          card: "var(--maude-card)",
          border: "var(--maude-border)",
          text: "var(--maude-text)",
          muted: "var(--maude-muted)",
          accent: "var(--maude-accent)",
          "accent-light": "var(--maude-accent-light)",
        },
      },
    },
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ["dark"],
  },
};
