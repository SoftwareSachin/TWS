module.exports = {
  parser: "@babel/eslint-parser",
  parserOptions: {
    requireConfigFile: false, // Avoid requiring a Babel config file
    babelOptions: {
      presets: ["next/babel"], // Use the Next.js Babel preset
    },
  },
  extends: [
    "next/core-web-vitals", // Next.js core web vitals linting
    "plugin:react/recommended", // Add React-specific linting rules
  ],
  plugins: ["react"], // Add the react plugin
  rules: {
    // Disable the `react/react-in-jsx-scope` rule for React 17+ (since React 17 doesn't need React in scope for JSX)
    "react/react-in-jsx-scope": "off",
    "react/prop-types": "off",
  },
};
