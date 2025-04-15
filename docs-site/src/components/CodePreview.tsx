import React, { useState } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-yaml';

interface CodeSection {
  title: string;
  code: string;
  isCollapsed?: boolean;
}

interface CodePreviewProps {
  code: string | CodeSection[];
  language: string;
  title?: string;
  showLineNumbers?: boolean;
  maxHeight?: number;
  defaultCollapsed?: boolean;
}

export default function CodePreview({
  code,
  language,
  title,
  showLineNumbers = true,
  maxHeight = 500,
  defaultCollapsed = false
}: CodePreviewProps) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(!defaultCollapsed);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  React.useEffect(() => {
    Prism.highlightAll();
  }, [code, isExpanded, expandedSections]);

  const handleCopy = async () => {
    const textToCopy = Array.isArray(code) 
      ? code.map(section => section.code).join('\n\n')
      : code;
    await navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleSection = (sectionTitle: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionTitle]: !prev[sectionTitle]
    }));
  };

  const renderCodeContent = () => {
    if (!isExpanded) {
      return (
        <div style={{
          padding: '1rem',
          color: '#666',
          textAlign: 'center',
          cursor: 'pointer'
        }} onClick={() => setIsExpanded(true)}>
          Click to expand code
        </div>
      );
    }

    if (Array.isArray(code)) {
      return (
        <div>
          {code.map((section, index) => (
            <div key={index} style={{ marginBottom: index < code.length - 1 ? '1rem' : 0 }}>
              <div
                onClick={() => toggleSection(section.title)}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: '#2d2d2d',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  borderBottom: '1px solid #3d3d3d'
                }}
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  style={{
                    transform: expandedSections[section.title] ? 'rotate(90deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s ease'
                  }}
                >
                  <path
                    d="M4 2L8 6L4 10"
                    stroke="#888"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <span style={{ color: '#e1e1e1', fontSize: '0.875rem' }}>
                  {section.title}
                </span>
              </div>
              {expandedSections[section.title] && (
                <pre style={{ margin: 0, padding: '1rem' }}>
                  <code className={`language-${language}`}>
                    {section.code}
                  </code>
                </pre>
              )}
            </div>
          ))}
        </div>
      );
    }

    return (
      <pre style={{ margin: 0, padding: '1rem' }}>
        <code className={`language-${language}`}>
          {code}
        </code>
      </pre>
    );
  };

  return (
    <div style={{
      backgroundColor: '#1e1e1e',
      borderRadius: '8px',
      overflow: 'hidden',
      marginBottom: '1rem'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0.5rem 1rem',
        backgroundColor: '#2d2d2d',
        borderBottom: '1px solid #3d3d3d'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {title && (
            <span style={{
              color: '#e1e1e1',
              fontSize: '0.875rem',
              fontFamily: 'monospace'
            }}>
              {title}
            </span>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              padding: '4px',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              style={{
                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease'
              }}
            >
              <path
                d="M4 6L8 10L12 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
        <div style={{
          display: 'flex',
          gap: '0.5rem',
          alignItems: 'center'
        }}>
          <span style={{
            color: '#888',
            fontSize: '0.75rem',
            textTransform: 'uppercase'
          }}>
            {language}
          </span>
          <button
            onClick={handleCopy}
            style={{
              backgroundColor: copied ? '#2ea043' : '#363636',
              border: 'none',
              borderRadius: '4px',
              padding: '4px 8px',
              color: '#fff',
              fontSize: '0.75rem',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            {copied ? (
              <>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M10 3L4.5 8.5L2 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M8 2H2V8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M4 4H10V10H4V4Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Copy
              </>
            )}
          </button>
        </div>
      </div>
      <div style={{
        maxHeight: isExpanded ? maxHeight : 'auto',
        overflow: 'auto'
      }}>
        {renderCodeContent()}
      </div>
    </div>
  );
}
