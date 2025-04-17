import React from 'react';
import Link from 'next/link';
import styles from '../styles/Home.module.css';

const Home: React.FC = () => {
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>Welcome to GoodRobot Documentation</h1>
        
        <div className={styles.grid}>
          <Link href="/getting-started" className={styles.card}>
            <h2>Getting Started</h2>
            <p>Learn how to set up and use the GoodRobot voice assistant.</p>
          </Link>

          <Link href="/api-docs" className={styles.card}>
            <h2>API Documentation</h2>
            <p>Explore the API endpoints and their functionality.</p>
          </Link>

          <Link href="/voice-interface" className={styles.card}>
            <h2>Voice Interface</h2>
            <p>Understand how to interact with the voice assistant.</p>
          </Link>

          <Link href="/system-architecture" className={styles.card}>
            <h2>System Architecture</h2>
            <p>Learn about the underlying architecture and components.</p>
          </Link>
        </div>
      </main>
    </div>
  );
};

export default Home;
