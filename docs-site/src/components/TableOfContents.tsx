import React, { useState, useEffect } from 'react';
import styles from '../styles/TableOfContents.module.css';

interface Subsection {
  id: string;
  title: string;
}

interface Section {
  id: string;
  title: string;
  subsections?: Subsection[];
}

interface TableOfContentsProps {
  sections: Section[];
}

const TableOfContents: React.FC<TableOfContentsProps> = ({ sections }) => {
  const [activeSection, setActiveSection] = useState<string>('');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: '-20% 0px -80% 0px' }
    );

    sections.forEach((section) => {
      const element = document.getElementById(section.id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, [sections]);

  const handleClick = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  return (
    <nav className={styles.toc}>
      <h2 className={styles.tocTitle}>Contents</h2>
      <ul className={styles.tocList}>
        {sections.map((section) => (
          <li key={section.id} className={styles.tocItem}>
            <div className={styles.tocSection}>
              <a
                href={`#${section.id}`}
                className={`${styles.tocLink} ${activeSection === section.id ? styles.active : ''}`}
                onClick={(e) => {
                  e.preventDefault();
                  handleClick(section.id);
                }}
              >
                {section.title}
              </a>
              {section.subsections && section.subsections.length > 0 && (
                <button
                  className={styles.toggleButton}
                  onClick={() => toggleSection(section.id)}
                  aria-expanded={expandedSections.has(section.id)}
                >
                  {expandedSections.has(section.id) ? '−' : '+'}
                </button>
              )}
            </div>
            {section.subsections && section.subsections.length > 0 && expandedSections.has(section.id) && (
              <ul className={styles.tocSubList}>
                {section.subsections.map((subsection) => (
                  <li key={subsection.id} className={styles.tocItem}>
                    <a
                      href={`#${subsection.id}`}
                      className={`${styles.tocLink} ${activeSection === subsection.id ? styles.active : ''}`}
                      onClick={(e) => {
                        e.preventDefault();
                        handleClick(subsection.id);
                      }}
                    >
                      {subsection.title}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default TableOfContents; 