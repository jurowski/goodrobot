import React from 'react';
import styles from '../styles/DocPage.module.css';

const ApiDocs: React.FC = () => {
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>API Documentation</h1>
        
        <section className={styles.section}>
          <h2>Endpoints</h2>
          <div className={styles.endpoint}>
            <h3>POST /voice</h3>
            <p>Process voice commands and return responses</p>
            <pre className={styles.code}>
              {`{
  "command": "string",
  "context": "object"
}`}
            </pre>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Authentication</h2>
          <p>Learn how to authenticate your API requests:</p>
          <ul>
            <li>API keys</li>
            <li>Token management</li>
            <li>Rate limiting</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Error Handling</h2>
          <p>Understand API error responses:</p>
          <ul>
            <li>Error codes</li>
            <li>Response formats</li>
            <li>Retry strategies</li>
          </ul>
        </section>
      </main>
    </div>
  );
};

export default ApiDocs; 