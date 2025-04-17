import React from 'react';
import TableOfContents from '../components/TableOfContents';
import styles from '../styles/GettingStarted.module.css';

const sections = [
  {
    id: 'installation',
    title: 'Installation',
    subsections: [
      { id: 'prerequisites', title: 'Prerequisites' },
      { id: 'setup', title: 'Setup Process' },
      { id: 'verification', title: 'Verification' }
    ]
  },
  {
    id: 'basic-usage',
    title: 'Basic Usage',
    subsections: [
      { id: 'configuration', title: 'Configuration' },
      { id: 'commands', title: 'Common Commands' },
      { id: 'examples', title: 'Examples' }
    ]
  },
  {
    id: 'configuration',
    title: 'Configuration',
    subsections: [
      { id: 'settings', title: 'Settings Overview' },
      { id: 'customization', title: 'Customization Options' },
      { id: 'advanced', title: 'Advanced Configuration' }
    ]
  }
];

const GettingStarted: React.FC = () => {
  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        <TableOfContents sections={sections} />
      </aside>
      <main className={styles.content}>
        <h1>Getting Started</h1>
        
        <section id="installation">
          <h2>Installation</h2>
          
          <section id="prerequisites">
            <h3>Prerequisites</h3>
            <p>Before you begin, ensure you have the following installed:</p>
            <ul>
              <li>Node.js (version 14 or higher)</li>
              <li>npm or yarn package manager</li>
              <li>Git (optional, but recommended)</li>
            </ul>
          </section>
          
          <section id="setup">
            <h3>Setup Process</h3>
            <p>Follow these steps to set up the project:</p>
            <ol>
              <li>Clone the repository or download the source code</li>
              <li>Navigate to the project directory</li>
              <li>Install dependencies using npm or yarn</li>
              <li>Configure your environment variables</li>
            </ol>
          </section>
          
          <section id="verification">
            <h3>Verification</h3>
            <p>To verify your installation:</p>
            <ol>
              <li>Run the test suite</li>
              <li>Start the development server</li>
              <li>Access the application in your browser</li>
            </ol>
          </section>
        </section>
        
        <section id="basic-usage">
          <h2>Basic Usage</h2>
          
          <section id="configuration">
            <h3>Configuration</h3>
            <p>Learn how to configure the basic settings:</p>
            <ul>
              <li>Environment variables</li>
              <li>Application settings</li>
              <li>User preferences</li>
            </ul>
          </section>
          
          <section id="commands">
            <h3>Common Commands</h3>
            <p>Essential commands for development:</p>
            <ul>
              <li><code>npm run dev</code> - Start development server</li>
              <li><code>npm run build</code> - Build for production</li>
              <li><code>npm run test</code> - Run test suite</li>
            </ul>
          </section>
          
          <section id="examples">
            <h3>Examples</h3>
            <p>Here are some common usage examples:</p>
            <ul>
              <li>Basic setup and configuration</li>
              <li>Creating your first component</li>
              <li>Implementing common features</li>
            </ul>
          </section>
        </section>
        
        <section id="configuration">
          <h2>Configuration</h2>
          
          <section id="settings">
            <h3>Settings Overview</h3>
            <p>Learn about the available configuration options:</p>
            <ul>
              <li>Application settings</li>
              <li>User preferences</li>
              <li>System configurations</li>
            </ul>
          </section>
          
          <section id="customization">
            <h3>Customization Options</h3>
            <p>Customize the application to your needs:</p>
            <ul>
              <li>Themes and styling</li>
              <li>Component behavior</li>
              <li>Feature toggles</li>
            </ul>
          </section>
          
          <section id="advanced">
            <h3>Advanced Configuration</h3>
            <p>Advanced configuration options:</p>
            <ul>
              <li>Performance optimization</li>
              <li>Security settings</li>
              <li>Integration options</li>
            </ul>
          </section>
        </section>
      </main>
    </div>
  );
};

export default GettingStarted; 