import React from 'react';

export function highlightText(text: string, searchTerm: string): React.ReactElement {
  if (!searchTerm) return <>{text}</>;
  
  const parts = text.split(new RegExp(`(${searchTerm})`, 'gi'));
  
  return (
    <>
      {parts.map((part, i) => (
        part.toLowerCase() === searchTerm.toLowerCase() ? (
          <mark key={i} style={{
            backgroundColor: '#fff3cd',
            padding: '0.1em 0',
            borderRadius: '2px',
            color: '#856404'
          }}>{part}</mark>
        ) : part
      ))}
    </>
  );
}
