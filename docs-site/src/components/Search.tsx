import React, { useState, useEffect, useMemo } from 'react';
import { useDocumentation } from '../context/DocumentationContext';
import { DocumentationSystem } from '../services/documentation';
import { useRouter } from 'next/router';
import { highlightText } from '../utils/highlightText';
import styles from '../styles/SearchResults.module.css';

interface Link {
  title: string;
  url: string;
}

interface Heuristic {
  title: string;
  description: string;
  links: Link[];
}

interface AcademicPaper {
  title: string;
  authors: string[];
  year: number;
  url: string;
  source: string;
}

interface SystemDetails {
  academicFoundations: AcademicPaper[];
  heuristics: Heuristic[];
  implementation: {
    description: string;
    codeExamples: Array<{
      title: string;
      language: string;
      code: string;
    }>;
  };
}

interface Category {
  id: string;
  name: string;
}

export default function Search() {
  const router = useRouter();
  const { documentation, loading, error } = useDocumentation();
  const [query, setQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const systems = useMemo(() => documentation?.systems || [], [documentation]);
  const categories = useMemo(() => documentation?.categories || [], [documentation]);

  const handleSearch = (searchQuery: string) => {
    setQuery(searchQuery);
    
    if (searchQuery.trim() && !searchHistory.includes(searchQuery.trim())) {
      setSearchHistory(prev => [searchQuery.trim(), ...prev].slice(0, 5));
    }
  };

  const filteredResults = useMemo(() => {
    if (!systems.length) return [];
    
    return systems.filter((system: DocumentationSystem) => {
      if (selectedCategory && system.category !== selectedCategory) return false;
      
      const matchesQuery = !query || 
        system.title.toLowerCase().includes(query.toLowerCase()) ||
        system.content.toLowerCase().includes(query.toLowerCase()) ||
        system.tags?.some((tag: string) => tag.toLowerCase().includes(query.toLowerCase()));
      
      return matchesQuery;
    });
  }, [systems, query, selectedCategory]);

  const getMatchingHeuristics = (system: DocumentationSystem): Heuristic[] => {
    return system.details?.heuristics.filter(heuristic => 
      heuristic.title.toLowerCase().includes(query.toLowerCase()) ||
      heuristic.description.toLowerCase().includes(query.toLowerCase())
    ) || [];
  };

  const renderHeuristicLinks = (heuristic: Heuristic): React.ReactElement => {
    return (
      <div className={styles.heuristicLinks}>
        <h4>Educational Resources:</h4>
        <ul>
          {heuristic.links.map((link, index) => (
            <li key={index}>
              <a 
                href={link.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className={styles.educationalLink}
              >
                {link.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderSearchResults = () => {
    if (!filteredResults.length) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: '#666'
        }}>
          No results found
        </div>
      );
    }

    return filteredResults.map((result: DocumentationSystem) => (
      <div
        key={result.id}
        data-testid="search-result"
        onClick={() => router.push(`/system/${result.id}`)}
        className={styles.searchResult}
      >
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '12px'
        }}>
          <h3 style={{
            margin: 0,
            fontSize: '1.25rem',
            color: '#1a1a1a'
          }}>
            {highlightText(result.title, query)}
          </h3>
        </div>
        <p style={{
          margin: '8px 0',
          color: '#666',
          fontSize: '0.9rem'
        }}>
          {highlightText(result.content, query)}
        </p>
        {result.tags && result.tags.length > 0 && (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '8px',
            marginTop: '8px'
          }}>
            {result.tags.map((tag, index) => (
              <span
                key={index}
                style={{
                  backgroundColor: '#f0f0f0',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  color: '#666'
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    ));
  };

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
        Loading documentation...
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
        <h2>Error loading documentation</h2>
        <p>{error.message}</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.searchContainer}>
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="Search documentation..."
          className={styles.searchInput}
        />
        <div className={styles.categoryFilter}>
          <select
            value={selectedCategory || ''}
            onChange={(e) => setSelectedCategory(e.target.value || null)}
            className={styles.categorySelect}
          >
            <option value="">All Categories</option>
            {categories.map((category: Category) => (
              <option key={category.id} value={category.id}>
                {category.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className={styles.searchResults}>
        {renderSearchResults()}
      </div>
    </div>
  );
} 