import React from 'react';
import styles from '../styles/DocPage.module.css';

const SystemArchitecture: React.FC = () => {
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>System Architecture</h1>
        
        <section className={styles.section}>
          <h2>Components</h2>
          <p>Key system components:</p>
          <ul>
            <li>Voice Interface Module</li>
            <li>API Server</li>
            <li>Task Management System</li>
            <li>Response Generation Engine</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Data Flow</h2>
          <p>Understand how data moves through the system:</p>
          <ul>
            <li>Voice input processing</li>
            <li>Command interpretation</li>
            <li>Response generation</li>
            <li>Output delivery</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Integration Points</h2>
          <p>System integration capabilities:</p>
          <ul>
            <li>External API connections</li>
            <li>Database integration</li>
            <li>Third-party service hooks</li>
          </ul>
        </section>
      </main>
    </div>
  );
};

export default SystemArchitecture; 