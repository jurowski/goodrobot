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
        <h2>Error loading system details</h2>
        <p>{error.message}</p>
      </div>
    );
  }

  const system = documentation?.systems.find((s: DocumentationSystem) => s.id === id) as DocumentationSystem | undefined;

  if (!system) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '4rem',
        gap: '1rem'
      }}>
        <h2>System not found</h2>
        <Link href="/test-search" style={{
          color: '#0066cc',
          textDecoration: 'none'
        }}>
          ← Back to search
        </Link>
      </div>
    );
  }

  return (
    <div style={{
      maxWidth: '1000px',
      margin: '0 auto',
      padding: '40px 20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        marginBottom: '2rem'
      }}>
        <Link href="/test-search" style={{
          color: '#0066cc',
          textDecoration: 'none',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          ← Back to search
        </Link>
      </div>

      <div style={{
        backgroundColor: 'white',
        borderRadius: '16px',
        padding: '2rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
        marginBottom: '2rem'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '1rem'
        }}>
          <h1 style={{
            margin: 0,
            color: '#1a1a1a',
            fontSize: '2.5rem'
          }}>{system.title}</h1>
          <span style={{
            backgroundColor: '#e9ecef',
            padding: '0.5rem 1rem',
            borderRadius: '20px',
            fontSize: '0.875rem',
            color: '#495057'
          }}>{system.category}</span>
        </div>

        <p style={{
          fontSize: '1.125rem',
          lineHeight: '1.7',
          color: '#4a5568',
          marginBottom: '2rem'
        }}>{system.content}</p>

        <div style={{
          display: 'flex',
          gap: '0.5rem',
          flexWrap: 'wrap',
          marginBottom: '2rem'
        }}>
          {system.tags?.map(tag => (
            <span key={tag} style={{
              backgroundColor: '#f1f3f5',
              padding: '0.25rem 0.75rem',
              borderRadius: '16px',
              fontSize: '0.875rem',
              color: '#666'
            }}>{tag}</span>
          ))}
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '2rem',
        marginBottom: '2rem'
      }}>
        <div style={{
          backgroundColor: 'white',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
        }}>
          <h2 style={{
            color: '#1a1a1a',
            fontSize: '1.5rem',
            marginBottom: '1rem'
          }}>Academic Foundations</h2>
          <ul style={{
            listStyle: 'none',
            padding: 0,
            margin: 0
          }}>
            {system.details?.academicFoundations.map((paper, index) => (
              <li key={index} style={{
                padding: '0.75rem',
                borderBottom: '1px solid #eee',
                color: '#4a5568'
              }}>
                <div style={{ marginBottom: '0.5rem' }}>
                  <a 
                    href={paper.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{
                      color: '#0066cc',
                      textDecoration: 'none',
                      fontWeight: '500'
                    }}
                  >
                    {paper.title}
                  </a>
                </div>
                <div style={{ fontSize: '0.875rem', color: '#666' }}>
                  <div>{paper.authors.join(', ')}</div>
                  <div>{paper.source}, {paper.year}</div>
                  <div>DOI: {paper.doi}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div style={{
          backgroundColor: 'white',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
        }}>
          <h2 style={{
            color: '#1a1a1a',
            fontSize: '1.5rem',
            marginBottom: '1rem'
          }}>Research-based Heuristics</h2>
          <ul style={{
            listStyle: 'none',
            padding: 0,
            margin: 0
          }}>
            {system.details?.heuristics.map((heuristic, index) => (
              <li key={index} style={{
                padding: '0.75rem',
                borderBottom: '1px solid #eee',
                color: '#4a5568'
              }}>
                <div style={{ marginBottom: '0.5rem' }}>
                  <h3 style={{
                    margin: 0,
                    fontSize: '1.1rem',
                    color: '#1a1a1a'
                  }}>
                    {heuristic.title}
                  </h3>
                  <p style={{
                    margin: '0.5rem 0',
                    fontSize: '0.875rem'
                  }}>
                    {heuristic.description}
                  </p>
                </div>
                {heuristic.links && heuristic.links.length > 0 && (
                  <div style={{ marginTop: '0.5rem' }}>
                    <h4 style={{
                      margin: '0.5rem 0',
                      fontSize: '0.875rem',
                      color: '#666'
                    }}>
                      Educational Resources:
                    </h4>
                    <ul style={{
                      listStyle: 'none',
                      padding: 0,
                      margin: 0
                    }}>
                      {heuristic.links.map((link, linkIndex) => (
                        <li key={linkIndex} style={{ marginBottom: '0.25rem' }}>
                          <a 
                            href={link.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            style={{
                              color: '#0066cc',
                              textDecoration: 'none',
                              fontSize: '0.875rem'
                            }}
                          >
                            {link.title}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {heuristic.relationships && heuristic.relationships.length > 0 && (
                  <div className={styles.relatedHeuristics}>
                    <h4>Related Heuristics</h4>
                    {heuristic.relationships.map((relationship: Relationship, relIndex: number) => (
                      <div key={relIndex} className={styles.relatedHeuristic}>
                        <h5>{relationship.heuristic}</h5>
                        <p>{relationship.description}</p>
                        <span className={styles.systemTag}>{relationship.system}</span>
                      </div>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div style={{
        backgroundColor: 'white',
        borderRadius: '16px',
        padding: '2rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
      }}>
        <h2 style={{
          color: '#1a1a1a',
          fontSize: '1.5rem',
          marginBottom: '1rem'
        }}>Implementation Details</h2>
        
        {system.details?.implementation.codeExamples.map((example, index) => {
          const sections = example.code.includes('// SECTION:') 
            ? example.code.split('// SECTION:').map(section => {
                const [title, ...codeLines] = section.trim().split('\n');
                return {
                  title: title || 'Main Code',
                  code: codeLines.join('\n'),
                  isCollapsed: true
                };
              })
            : null;

          return (
            <CodePreview
              key={index}
              code={sections || example.code}
              language={example.language}
              title={example.title}
              showLineNumbers={true}
              defaultCollapsed={example.code.split('\n').length > 20}
              maxHeight={600}
            />
          );
        })}
        
        {system.details?.implementation.description && (
          <p style={{
            color: '#4a5568',
            marginTop: '1rem'
          }}>
            {system.details.implementation.description}
          </p>
        )}
      </div>

      <div style={{
        marginTop: '2rem',
        color: '#718096',
        fontSize: '0.875rem',
        textAlign: 'right'
      }}>
        Last updated: {system.lastUpdated}
      </div>
    </div>
  );
}
