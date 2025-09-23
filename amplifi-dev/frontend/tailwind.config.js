/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "select-option": "#374AF1",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          1: "hsl(var(--chart-1))",
          2: "hsl(var(--chart-2))",
          3: "hsl(var(--chart-3))",
          4: "hsl(var(--chart-4))",
          5: "hsl(var(--chart-5))",
        },
        black: {
          10: "#001529",
          20: "#292929",
        },
        gray: {
          10: "#FCFDFD",
          20: "#E2EDF7",
          30: "#2929297A",
        },
        blue: {
          10: "#374AF1",
          20: "#EFF1FE",
        },
        green: {
          10: "#64AC01",
          20: "#386D07",
        },
        custom: {
          blueBg: "#D1DFFA",
          blueText: "#0A255C",
          redBg: "#FAD1D1",
          redText: "#5C0A0A",
          pinkBg: "#FAD1F3",
          pinkText: "#5C0A4E",
          tealBg: "#D1F3FA",
          tealText: "#0A4E5C",
          yellowBg: "#FAFAD1",
          yellowText: "#5C5C0A",
          purpleBg: "#ECD1FA",
          purpleText: "#410A5C",
          customBlue: "#374AF1",
          Danger: "#992B3C",
          headerColor: "#eff1fe",
          Success: "#386D07",
          Processing: "#0A255C",
          toolColor: "#ABABAB",
          contextBgColor: "#F8FAFC",
          contextHoverButtonColor: "#EFF1FE",
          tableHeader: "#D7DBFC",
          tableColumn: "#292929F5",
          numberColor: "#EBEDFE",
          warning: "#9E6E00",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontSize: {
        "3.2xl": "32px",
        "6.5xl": ["64px", "72px"],
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  safelist: [
    "w-[500px]",
    {
      pattern:
        /^(border|bg|text|ring|outline|shadow|fill|stroke)-select-option$/,
    },
  ],
  plugins: [require("tailwindcss-animate")],
};
