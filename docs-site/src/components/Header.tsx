import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import styles from '../styles/Header.module.css';

const Header: React.FC = () => {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <div className={styles.brainIcon}>
          <Image
            src="/images/light-mode-brain-2.png"
            alt="Brain Icon"
            width={120}
            height={120}
          />
        </div>
        <div className={styles.logo}>
          <Link href="/">GoodRobot</Link>
        </div>
      </div>
      <nav className={styles.nav}>
        <ul>
          <li>
            <Link href="/">Home</Link>
          </li>
          <li>
            <a href="http://localhost:8000/redoc" target="_blank" rel="noopener noreferrer">API Docs</a>
          </li>
          <li>
            <a href="http://localhost:8000/voice" target="_blank" rel="noopener noreferrer">Voice Assistant</a>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 