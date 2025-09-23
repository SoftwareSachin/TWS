// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Amplifi Documentation',
  tagline: 'Get your unstructured data ready for Gen AI Applications',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://dev-docs.amplifi.com',
  baseUrl: '/',

  // Organization details
  organizationName: 'thoughtswinsystems',
  projectName: 'amplifi-docs',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/', // Serve docs at the root
          // editUrl:
          //   'https://github.com/thoughtswinsystems/amplifi-docs/tree/main/docs/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/amplifi-social-card.jpg',
      navbar: {
        title: '',
        logo: {
          alt: 'Amplifi Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docs',  
            position: 'left',
            label: 'Documentation',  
          },
//          {
//            href: 'https://dev-app.dataamplifi.com',
//            label: 'Product Website',
//            position: 'right',
//          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/getting-started/account-setup',
              },
              {
                label: 'Core Concepts',
                to: '/core-concepts/data-sources',
              },
              {
                label: 'How-To Guides',
                to: '/how-to-guides/connecting-data-sources',
              },
            ],
          },
          {
            title: 'Company',
            items: [
              {
                label: 'ThoughtsWin Systems',
                href: 'https://thoughtswinsystems.com/',
              },
              {
                label: 'Amplifi Platform',
                href: 'https://thoughtswinsystems.com/amplifi/',
              },
              {
                label: 'Contact',
                href: 'https://thoughtswinsystems.com/contact/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} ThoughtsWin Systems. All rights reserved.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
