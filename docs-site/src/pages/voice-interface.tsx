import React from 'react';
import styles from '../styles/DocPage.module.css';

const VoiceInterface: React.FC = () => {
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>Voice Interface</h1>
        
        <section className={styles.section}>
          <h2>Wake Word</h2>
          <p>Learn how to use the wake word feature:</p>
          <ul>
            <li>Default wake word: "Hey Robot"</li>
            <li>Custom wake word configuration</li>
            <li>Wake word sensitivity settings</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Voice Commands</h2>
          <p>Available voice commands:</p>
          <ul>
            <li>Basic commands (start, stop, help)</li>
            <li>Task management commands</li>
            <li>System control commands</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Response Types</h2>
          <p>Understand different response formats:</p>
          <ul>
            <li>Text responses</li>
            <li>Audio responses</li>
            <li>Visual feedback</li>
          </ul>
        </section>
      </main>
    </div>
  );
};

export default VoiceInterface; 