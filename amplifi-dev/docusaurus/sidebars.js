// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/account-setup', 'getting-started/first-workspace'/*, 'getting-started/navigation-basics'*/],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: ['core-concepts/data-sources', 'core-concepts/ingestion', 'core-concepts/destinations','core-concepts/knowledge-graph','core-concepts/system-tools','core-concepts/mcp-tool','core-concepts/agentic-ai'],
    },
    {
      type: 'category',
      label: 'How-To Guides',
      items: [ 'how-to-guides/datasets','how-to-guides/connecting-data-sources', 'how-to-guides/add-new-users',  'how-to-guides/configuring-ingestion','how-to-guides/creating-graph', 'how-to-guides/search-dataset','how-to-guides/connecting-destination','how-to-guides/create-workflow','how-to-guides/creating-tool','how-to-guides/configuring-mcp','how-to-guides/creating-agent', 'how-to-guides/creating-chatapp']
    }
  ],
};

module.exports = sidebars;
