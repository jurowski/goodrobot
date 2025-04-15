import React, { createContext, useContext, useState, useEffect } from 'react';
import { Documentation, loadDocumentation } from '../services/documentation';

interface DocumentationContextType {
  documentation: Documentation | null;
  loading: boolean;
  error: Error | null;
}

const DocumentationContext = createContext<DocumentationContextType>({
  documentation: null,
  loading: true,
  error: null
});

export function DocumentationProvider({ children }: { children: React.ReactNode }) {
  const [documentation, setDocumentation] = useState<Documentation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    loadDocumentation()
      .then(data => {
        setDocumentation(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, []);

  return (
    <DocumentationContext.Provider value={{ documentation, loading, error }}>
      {children}
    </DocumentationContext.Provider>
  );
}

export function useDocumentation() {
  return useContext(DocumentationContext);
}
