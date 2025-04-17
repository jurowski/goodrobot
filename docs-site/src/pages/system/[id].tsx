import React from 'react';
import { useRouter } from 'next/router';
import { useDocumentation } from '../../context/DocumentationContext';
import Link from 'next/link';
import CodePreview from '../../components/CodePreview';
import { DocumentationSystem, Heuristic } from '../../services/documentation';
import styles from '../../styles/SystemDetails.module.css';

interface Link {
  title: string;
  url: string;
  type: 'wikipedia' | 'academic' | 'tutorial';
}

interface Relationship {
  heuristic: string;
  description: string;
  system: string;
}

export default function SystemDetail() {
  const router = useRouter();
  const { id } = router.query;
  const { documentation, loading, error } = useDocumentation();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        fontSize: '1.2rem',
        color: '#666'
      }}>
        Loading system details...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        color: '#dc3545',
        flexDirection: 'column',
        gap: '1rem'
      }}>
        Error loading system details
        <button onClick={() => router.reload()}>Retry</button>
      </div>
    );
  }

  const system = documentation?.systems.find(s => s.id === id);

  if (!system) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        color: '#666'
      }}>
        System not found
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>{system.title}</h1>
        <div className={styles.metadata}>
          <span className={styles.category}>{system.category}</span>
          <span className={styles.score}>Score: {system.score}</span>
          {system.lastUpdated && (
            <span className={styles.lastUpdated}>
              Last updated: {new Date(system.lastUpdated).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>

      <div className={styles.content}>
        <div className={styles.mainContent}>
          <p>{system.content}</p>

          {system.details && (
            <>
              {system.details.heuristics && system.details.heuristics.length > 0 && (
                <div className={styles.section}>
                  <h2>Key Features</h2>
                  {system.details.heuristics.map((heuristic, index) => (
                    <div key={index} className={styles.heuristic}>
                      <h3>{heuristic.title}</h3>
                      <p>{heuristic.description}</p>
                      {heuristic.links && heuristic.links.length > 0 && (
                        <div className={styles.links}>
                          {heuristic.links.map((link, linkIndex) => (
                            <a
                              key={linkIndex}
                              href={link.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={styles.link}
                            >
                              {link.title}
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {system.details.implementation && (
                <div className={styles.section}>
                  <h2>Implementation</h2>
                  <p>{system.details.implementation.description}</p>
                  {system.details.implementation.codeExamples.map((example, index) => (
                    <CodePreview
                      key={index}
                      title={example.title}
                      language={example.language}
                      code={example.code}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        <div className={styles.sidebar}>
          {system.tags && system.tags.length > 0 && (
            <div className={styles.tags}>
              <h3>Tags</h3>
              <div className={styles.tagList}>
                {system.tags.map((tag, index) => (
                  <span key={index} className={styles.tag}>
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
